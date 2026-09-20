"""Evaluate the trained SFT LoRA adapter on the GSM8K test split."""

import argparse
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Support both direct script execution and python -m evaluation.sft_eval.
if __package__:
    from .base_eval import evaluate, logger
else:
    from base_eval import evaluate, logger

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen2.5-0.5B"
ADAPTER_DIR = ROOT / "output/sft/final_adapter"
DATA_FILE = ROOT / "dataset/test.jsonl"
OUTPUT_DIR = ROOT / "output/sft_eval"
MAX_PROMPT_LENGTH = 512
BATCH_SIZE = 16
MAX_NEW_TOKENS = 256
MAX_SAMPLES = None  # Set to 8 for a quick debugging run.
SEED = 42


def load_model_and_tokenizer(model_path):
    if not (ADAPTER_DIR / "adapter_config.json").exists():
        raise FileNotFoundError(f"SFT adapter not found: {ADAPTER_DIR}")

    adapter_config = PeftConfig.from_pretrained(ADAPTER_DIR)
    if adapter_config.base_model_name_or_path != model_path:
        raise ValueError(
            f"Adapter expects base model {adapter_config.base_model_name_or_path!r}, "
            f"but MODEL is {model_path!r}. Set MODEL to the training base model "
            "in both evaluation scripts."
        )

    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()

    if use_bf16:
        dtype = torch.bfloat16
    elif use_cuda:
        dtype = torch.float16
    else:
        dtype = torch.float32

    logger.info("Loading SFT tokenizer: %s", ADAPTER_DIR)
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)

    logger.info("Loading SFT base model: %s | dtype=%s", model_path, dtype)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
    )

    logger.info("Loading SFT LoRA adapter: %s", ADAPTER_DIR)
    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_DIR,
        is_trainable=False,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.config.use_cache = True
    model.eval()

    logger.info("SFT model and adapter loaded")
    model_label = f"{model_path} + LoRA adapter {ADAPTER_DIR}"
    return model, tokenizer, model_label


def main():
    args = argparse.Namespace(
        stage="SFT",
        model_path=MODEL,
        adapter_path=ADAPTER_DIR,
        data_path=DATA_FILE,
        output_dir=OUTPUT_DIR,
        batch_size=BATCH_SIZE,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_new_tokens=MAX_NEW_TOKENS,
        max_samples=MAX_SAMPLES,
        seed=SEED,
    )
    evaluate(args, model_loader=load_model_and_tokenizer)


if __name__ == "__main__":
    main()
