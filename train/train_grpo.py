"""Train a GRPO LoRA adapter from the merged SFT model."""

import csv
import json
import logging
import re
import sys
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer, TrainerCallback, set_seed
from transformers.trainer_callback import PrinterCallback
from trl import GRPOConfig, GRPOTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "output/sft/merged_model"
TRAIN_FILE = ROOT / "dataset/train.jsonl"
OUTPUT_DIR = ROOT / "output/grpo_k8"
IMAGE_FILE = ROOT / "image/grpo_k8_training.jpg"

SEED = 42
MAX_COMPLETION_LENGTH = 256
EPOCHS = 1
BATCH_SIZE = 8
GRADIENT_ACCUMULATION = 2
NUM_GENERATIONS = 8
LEARNING_RATE = 5e-6
TEMPERATURE = 1.0
TOP_P = 1.0
BETA = 0.04
EPSILON = 0.2
LOGGING_STEPS = 1
PRINT_STEPS = 10
SAVE_STEPS = 100
MOVING_AVERAGE_WINDOW = 50
MAX_SAMPLES = None

LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.0
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

NUMBER_PATTERN = re.compile(
    r"[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?(?:\s*/\s*[-+]?\d[\d,]*)?"
)
FINAL_ANSWER_PATTERNS = (
    re.compile(r"final\s+answer\s*(?:is|=|:)\s*([^\n]+)", re.IGNORECASE),
    re.compile(r"answer\s*(?:is|=|:)\s*([^\n]+)", re.IGNORECASE),
)


def extract_last_boxed(text):
    starts = list(re.finditer(r"\\boxed\s*\{", text))
    for match in reversed(starts):
        content_start = match.end()
        depth = 1
        for index in range(content_start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    return text[content_start:index].strip()
    return None


def extract_answer(text):
    boxed = extract_last_boxed(text)
    if boxed is not None:
        return boxed

    for pattern in FINAL_ANSWER_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            numbers = NUMBER_PATTERN.findall(matches[-1])
            return numbers[-1] if numbers else matches[-1].strip().rstrip(".$")

    numbers = NUMBER_PATTERN.findall(text)
    return numbers[-1] if numbers else None


def to_fraction(value):
    if value is None:
        return None

    cleaned = value.strip()
    cleaned = re.sub(r"^\\(?:d)?frac\{([^{}]+)\}\{([^{}]+)\}$", r"\1/\2", cleaned)
    cleaned = re.sub(r"^\\text\{([^{}]+)\}$", r"\1", cleaned)
    cleaned = cleaned.replace("$", "").replace(",", "").replace("\\%", "%")
    cleaned = cleaned.rstrip(". ")
    if cleaned.endswith("%"):
        cleaned = cleaned[:-1].strip()

    try:
        if "/" in cleaned:
            numerator, denominator = cleaned.split("/", maxsplit=1)
            return Fraction(Decimal(numerator.strip())) / Fraction(Decimal(denominator.strip()))
        return Fraction(Decimal(cleaned))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None


def answers_equal(predicted, gold_answer):
    predicted_number = to_fraction(predicted)
    gold_number = to_fraction(gold_answer)
    if predicted_number is not None and gold_number is not None:
        return predicted_number == gold_number
    if predicted is None:
        return False
    return predicted.strip().lower() == gold_answer.strip().lower()


def correctness_reward(completions, gold_answer, **kwargs):
    """Return one for a numerically correct completion and zero otherwise."""
    return [
        float(answers_equal(extract_answer(completion), gold))
        for completion, gold in zip(completions, gold_answer)
    ]


def read_training_dataset(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            rows.append(
                {
                    "prompt": record["prompt"],
                    "gold_answer": record["gold_answer"],
                }
            )

    if MAX_SAMPLES is not None:
        rows = rows[:MAX_SAMPLES]
    if not rows:
        raise ValueError(f"Empty training dataset: {path}")

    logger.info("GRPO dataset loaded: %s | examples=%d", path, len(rows))
    return Dataset.from_list(rows)


def find_metric(record, key):
    candidates = (key, f"train_{key}", f"train/{key}")
    for candidate in candidates:
        if candidate in record:
            return record[candidate]
    return None


def moving_average(values, window):
    averages = []
    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += value
        if index >= window:
            running_sum -= values[index - window]
        count = min(index + 1, window)
        averages.append(running_sum / count)
    return averages


def enrich_history(history):
    enriched = []
    for original in history:
        record = dict(original)
        reward = find_metric(record, "reward")
        reward_std = find_metric(record, "reward_std")
        zero_std_rate = find_metric(record, "frac_reward_zero_std")

        if reward is not None:
            record["episode_reward"] = reward
        if reward_std is not None:
            record["episode_reward_std"] = reward_std
        if zero_std_rate is not None:
            record["mixed_group_rate"] = 1.0 - zero_std_rate

        enriched.append(record)
    return enriched


def save_training_artifacts(history):
    """Save per-step metrics and the reward, loss and KL training curves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = enrich_history(history)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    history_file = OUTPUT_DIR / "training_history.csv"
    fields = sorted({key for record in history for key in record})
    with history_file.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)

    reward_steps = []
    episode_rewards = []
    loss_steps = []
    policy_losses = []
    kl_steps = []
    kl_values = []

    for record in history:
        if "episode_reward" in record and "step" in record:
            reward_steps.append(record["step"])
            episode_rewards.append(record["episode_reward"])

        loss = find_metric(record, "loss")
        if loss is not None and "step" in record:
            loss_steps.append(record["step"])
            policy_losses.append(loss)

        kl = find_metric(record, "kl")
        if kl is not None and "step" in record:
            kl_steps.append(record["step"])
            kl_values.append(kl)

    if not episode_rewards:
        raise ValueError("No episode reward metrics were found in the training history")
    if not policy_losses:
        raise ValueError("No policy loss metrics were found in the training history")
    if not kl_values:
        raise ValueError("No KL metrics were found in the training history")

    reward_ma = moving_average(episode_rewards, MOVING_AVERAGE_WINDOW)
    figure, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    axes[0].plot(
        reward_steps,
        episode_rewards,
        color="#94a3b8",
        linewidth=0.8,
        alpha=0.55,
        label="Episode reward",
    )
    axes[0].plot(
        reward_steps,
        reward_ma,
        color="#2563eb",
        linewidth=2.2,
        label=f"Moving average ({MOVING_AVERAGE_WINDOW} steps)",
    )
    axes[0].set_ylabel("Reward")
    axes[0].set_title("GRPO episode reward")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(loss_steps, policy_losses, color="#7c3aed", linewidth=1.5)
    axes[1].set_ylabel("Policy loss")
    axes[1].set_title("GRPO policy loss")
    axes[1].grid(alpha=0.25)

    axes[2].plot(kl_steps, kl_values, color="#dc2626", linewidth=1.5)
    axes[2].set_xlabel("Optimizer step")
    axes[2].set_ylabel("KL divergence")
    axes[2].set_title("Policy divergence from SFT reference")
    axes[2].grid(alpha=0.25)

    IMAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(IMAGE_FILE, dpi=180, format="jpg")
    plt.close(figure)

    logger.info("Training history saved: %s", history_file)
    logger.info("GRPO plot saved: %s", IMAGE_FILE)


class GRPOLoggingCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            logger.info(
                "GRPO started | total_steps=%d | group_size=%d | print_every=%d",
                state.max_steps,
                NUM_GENERATIONS,
                PRINT_STEPS,
            )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or not logs:
            return
        if state.global_step % PRINT_STEPS != 0 and state.global_step != state.max_steps:
            return

        reward = find_metric(logs, "reward")
        reward_std = find_metric(logs, "reward_std")
        zero_std_rate = find_metric(logs, "frac_reward_zero_std")
        mixed_group_rate = None if zero_std_rate is None else 1.0 - zero_std_rate
        loss = find_metric(logs, "loss")
        kl = find_metric(logs, "kl")

        logger.info(
            "GRPO | step=%d/%d | reward=%s | reward_std=%s | mixed_group_rate=%s | policy_loss=%s | kl=%s",
            state.global_step,
            state.max_steps,
            f"{reward:.4f}" if reward is not None else "n/a",
            f"{reward_std:.4f}" if reward_std is not None else "n/a",
            f"{mixed_group_rate:.4f}" if mixed_group_rate is not None else "n/a",
            f"{loss:.6f}" if loss is not None else "n/a",
            f"{kl:.6f}" if kl is not None else "n/a",
        )

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            logger.info("Checkpoint saved: %s/checkpoint-%d", args.output_dir, state.global_step)


def main():
    if not (MODEL_DIR / "config.json").exists():
        raise FileNotFoundError(
            f"Merged SFT model not found: {MODEL_DIR}. Run python merge/merge_sft.py first."
        )

    effective_batch_size = BATCH_SIZE * GRADIENT_ACCUMULATION
    if effective_batch_size % NUM_GENERATIONS != 0:
        raise ValueError(
            "BATCH_SIZE * GRADIENT_ACCUMULATION must be divisible by NUM_GENERATIONS"
        )

    set_seed(SEED)
    train_dataset = read_training_dataset(TRAIN_FILE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    model_dtype = "bfloat16" if use_bf16 else ("float16" if use_cuda else "float32")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )

    training_args = GRPOConfig(
        output_dir=str(OUTPUT_DIR),
        model_init_kwargs={"torch_dtype": model_dtype},
        seed=SEED,
        data_seed=SEED,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_steps=0.03,
        lr_scheduler_type="cosine",
        max_completion_length=MAX_COMPLETION_LENGTH,
        num_generations=NUM_GENERATIONS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        beta=BETA,
        epsilon=EPSILON,
        loss_type="dapo",
        mask_truncated_completions=True,
        scale_rewards="group",
        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=use_bf16,
        fp16=use_cuda and not use_bf16,
        report_to="none",
        disable_tqdm=True,
        remove_unused_columns=False,
        use_vllm=False,
    )

    logger.info(
        "Loading merged SFT model and starting GRPO | model=%s | group_size=%d",
        MODEL_DIR,
        NUM_GENERATIONS,
    )
    trainer = GRPOTrainer(
        model=str(MODEL_DIR),
        reward_funcs=correctness_reward,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        callbacks=[GRPOLoggingCallback()],
    )
    trainer.remove_callback(PrinterCallback)
    train_result = trainer.train()

    adapter_dir = OUTPUT_DIR / "final_adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)
    trainer.save_state()

    if trainer.is_world_process_zero():
        save_training_artifacts(trainer.state.log_history)
        summary = dict(train_result.metrics)
        summary.update(
            {
                "base_model": str(MODEL_DIR),
                "train_examples": len(train_dataset),
                "num_generations": NUM_GENERATIONS,
                "reward": "numeric correctness only",
                "moving_average_window": MOVING_AVERAGE_WINDOW,
                "final_adapter": str(adapter_dir),
            }
        )
        with (OUTPUT_DIR / "training_summary.json").open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)
        logger.info("GRPO complete | adapter=%s", adapter_dir)


if __name__ == "__main__":
    main()
