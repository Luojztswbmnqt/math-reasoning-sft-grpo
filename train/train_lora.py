"""SFT LoRA training, validation, loss logging and plotting."""

import csv
import json
import logging
import sys
from functools import partial
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_callback import PrinterCallback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# File paths
ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen2.5-0.5B"
TRAIN_FILE = ROOT / "dataset/train.jsonl"
EVAL_FILE = ROOT / "dataset/eval.jsonl"
OUTPUT_DIR = ROOT / "output/sft"
IMAGE_FILE = ROOT / "image/sft_loss.png"

# Training parameters
SEED = 42
MAX_LENGTH = 512
EPOCHS = 3
BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 1e-4
LOGGING_STEPS = 1  # Log train loss after every optimizer step.
EVAL_STEPS = 100
SAVE_STEPS = 100

# LoRA parameters
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


class SFTDataset(Dataset):
    """Read JSONL data and compute loss only on response tokens."""

    def __init__(self, path, tokenizer):
        self.examples = []
        logger.info("Loading dataset: %s", path)

        with Path(path).open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue

                row = json.loads(line)
                prompt_ids = tokenizer.encode(
                    row["prompt"],
                    add_special_tokens=False,
                )
                response_ids = tokenizer.encode(
                    row["response"],
                    add_special_tokens=False,
                )
                response_ids.append(tokenizer.eos_token_id)

                # Reserve at least one token for the response.
                prompt_ids = prompt_ids[:MAX_LENGTH - 1]
                response_budget = MAX_LENGTH - len(prompt_ids)
                response_ids = response_ids[:response_budget]

                input_ids = prompt_ids + response_ids
                attention_mask = [1] * len(input_ids)
                labels = [-100] * len(prompt_ids) + response_ids.copy()

                example = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
                self.examples.append(example)

        if not self.examples:
            raise ValueError(f"Empty dataset: {path}")

        logger.info("Dataset loaded: %s | examples=%d", path, len(self.examples))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class ConsoleLoggingCallback(TrainerCallback):
    """Log training progress, train loss and eval loss to the console."""

    def __init__(self):
        self.previous_train_loss = None
        self.previous_eval_loss = None

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        effective_batch_size = (
            args.per_device_train_batch_size
            * args.gradient_accumulation_steps
            * args.world_size
        )
        logger.info(
            "Training started | epochs=%s | total_steps=%d | effective_batch_size=%d",
            args.num_train_epochs,
            state.max_steps,
            effective_batch_size,
        )
        logger.info(
            "Train loss every %d optimizer step(s); eval loss every %d optimizer step(s)",
            args.logging_steps,
            args.eval_steps,
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or not logs:
            return

        epoch = state.epoch if state.epoch is not None else 0.0

        if "loss" in logs:
            loss = logs["loss"]
            change = ""
            if self.previous_train_loss is not None:
                change = f" | change={loss - self.previous_train_loss:+.6f}"

            logger.info(
                "TRAIN | step=%d/%d | epoch=%.2f | loss=%.6f | lr=%.3e%s",
                state.global_step,
                state.max_steps,
                epoch,
                loss,
                logs.get("learning_rate", 0.0),
                change,
            )
            self.previous_train_loss = loss

        if "eval_loss" in logs:
            loss = logs["eval_loss"]
            change = ""
            if self.previous_eval_loss is not None:
                change = f" | change={loss - self.previous_eval_loss:+.6f}"

            logger.info(
                "EVAL  | step=%d/%d | epoch=%.2f | loss=%.6f | runtime=%.2fs%s",
                state.global_step,
                state.max_steps,
                epoch,
                loss,
                logs.get("eval_runtime", 0.0),
                change,
            )
            self.previous_eval_loss = loss

        if "train_loss" in logs:
            logger.info(
                "Training summary | mean_train_loss=%.6f | runtime=%.2fs",
                logs["train_loss"],
                logs.get("train_runtime", 0.0),
            )

    def on_step_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero and control.should_evaluate:
            logger.info("Starting evaluation | step=%d", state.global_step)

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            logger.info("Checkpoint saved: %s", checkpoint_dir)

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            logger.info("Training finished | completed_steps=%d", state.global_step)


def collate(features, pad_token_id):
    """Pad batch sequences to equal length and exclude padding from loss."""
    max_length = max(len(example["input_ids"]) for example in features)

    input_ids = []
    attention_masks = []
    labels = []

    for example in features:
        padding_length = max_length - len(example["input_ids"])

        input_ids.append(
            example["input_ids"] + [pad_token_id] * padding_length
        )
        attention_masks.append(
            example["attention_mask"] + [0] * padding_length
        )
        labels.append(
            example["labels"] + [-100] * padding_length
        )

    batch = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
    return batch


def save_artifacts(history):
    """Save training history and plot training and validation loss."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    history_file = OUTPUT_DIR / "training_history.csv"

    fieldnames = set()
    for record in history:
        fieldnames.update(record.keys())

    with history_file.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=sorted(fieldnames))
        writer.writeheader()
        writer.writerows(history)

    logger.info("Training history saved: %s", history_file)

    train_steps = []
    train_losses = []
    eval_steps = []
    eval_losses = []

    for record in history:
        if "loss" in record:
            train_steps.append(record["step"])
            train_losses.append(record["loss"])

        if "eval_loss" in record:
            eval_steps.append(record["step"])
            eval_losses.append(record["eval_loss"])

    # Very short runs may only contain the final mean training loss.
    if not train_losses:
        for record in history:
            if "train_loss" in record:
                train_steps.append(record["step"])
                train_losses.append(record["train_loss"])

    figure, axis = plt.subplots(figsize=(9, 5))

    if train_losses:
        axis.plot(train_steps, train_losses, label="Train loss")
    if eval_losses:
        axis.plot(eval_steps, eval_losses, label="Eval loss")

    axis.set_xlabel("Optimizer step")
    axis.set_ylabel("Loss")
    axis.set_title("GSM8K SFT LoRA")
    axis.grid(alpha=0.25)
    axis.legend()

    IMAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(IMAGE_FILE, dpi=180)
    plt.close(figure)
    logger.info("Loss plot saved: %s", IMAGE_FILE)


def main():
    set_seed(SEED)

    if not TRAIN_FILE.exists() or not EVAL_FILE.exists():
        raise FileNotFoundError("Run python prepare_dataset.py first")

    # 1. Load the tokenizer and datasets.
    logger.info("Loading tokenizer: %s", MODEL)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    logger.info("Tokenizer loaded")

    train_dataset = SFTDataset(TRAIN_FILE, tokenizer)
    eval_dataset = SFTDataset(EVAL_FILE, tokenizer)

    # 2. Select precision for the device and load the base model.
    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    use_fp16 = use_cuda and not use_bf16

    if use_bf16:
        dtype = torch.bfloat16
    elif use_fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    logger.info("Loading model: %s | CUDA=%s | dtype=%s", MODEL, use_cuda, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=dtype,
    )
    model.config.use_cache = False
    logger.info("Model loaded")

    # 3. Add LoRA adapters and freeze the base model weights.
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    trainable_parameters, total_parameters = model.get_nb_trainable_parameters()
    logger.info(
        "LoRA ready | rank=%d | target_modules=%s | trainable_parameters=%s/%s (%.2f%%)",
        LORA_RANK,
        ", ".join(LORA_TARGET_MODULES),
        f"{trainable_parameters:,}",
        f"{total_parameters:,}",
        100 * trainable_parameters / total_parameters,
    )

    # 4. Configure the Trainer.
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        seed=SEED,
        data_seed=SEED,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_steps=0.03,
        lr_scheduler_type="cosine",
        logging_steps=LOGGING_STEPS,
        logging_strategy="steps",
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=use_bf16,
        fp16=use_fp16,
        report_to="none",
        disable_tqdm=True,
        remove_unused_columns=False,
        optim="adamw_torch",
    )

    data_collator = partial(
        collate,
        pad_token_id=tokenizer.pad_token_id,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[ConsoleLoggingCallback()],
    )
    # Use formatted logs instead of duplicate output from the default callback.
    trainer.remove_callback(PrinterCallback)

    # 5. Train and compute the final validation loss.
    train_result = trainer.train()
    logger.info("Starting final evaluation")
    eval_metrics = trainer.evaluate()

    # 6. Save adapters, training state, metrics and the loss plot.
    adapter_dir = OUTPUT_DIR / "final_adapter"
    logger.info("Saving adapter and training state: %s", adapter_dir)
    trainer.save_model(str(adapter_dir))
    trainer.save_state()

    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(adapter_dir)

        metrics = dict(train_result.metrics)
        metrics.update(eval_metrics)
        metrics.update({
            "model": MODEL,
            "seed": SEED,
            "train_examples": len(train_dataset),
            "eval_examples": len(eval_dataset),
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "lora_target_modules": LORA_TARGET_MODULES,
            "trainable_parameters": trainable_parameters,
            "total_parameters": total_parameters,
            "trainable_percentage": 100 * trainable_parameters / total_parameters,
        })

        summary_file = OUTPUT_DIR / "training_summary.json"
        with summary_file.open("w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2)
        logger.info("Training summary saved: %s", summary_file)

        save_artifacts(trainer.state.log_history)

        logger.info("All done | adapter=%s | loss_plot=%s", adapter_dir, IMAGE_FILE)


if __name__ == "__main__":
    main()
