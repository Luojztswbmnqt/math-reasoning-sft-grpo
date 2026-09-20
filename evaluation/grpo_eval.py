"""Evaluate the GRPO LoRA adapter on the GSM8K test split."""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

if __package__:
    from .base_eval import evaluate, logger
else:
    from base_eval import evaluate, logger

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "output/sft/merged_model"
ADAPTER_DIR = ROOT / "output/grpo/final_adapter"
DATA_FILE = ROOT / "dataset/test.jsonl"
OUTPUT_DIR = ROOT / "output/grpo_eval"
MAX_PROMPT_LENGTH = 512
BATCH_SIZE = 16
MAX_NEW_TOKENS = 256
MAX_SAMPLES = None
SEED = 42


def load_model_and_tokenizer(model_path):
    model_path = Path(model_path)
    if not (model_path / "config.json").exists():
        raise FileNotFoundError(f"Merged SFT model not found: {model_path}")
    if not (ADAPTER_DIR / "adapter_config.json").exists():
        raise FileNotFoundError(f"GRPO adapter not found: {ADAPTER_DIR}")

    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    if use_bf16:
        dtype = torch.bfloat16
    elif use_cuda:
        dtype = torch.float16
    else:
        dtype = torch.float32

    logger.info("Loading GRPO tokenizer: %s", ADAPTER_DIR)
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)

    logger.info("Loading merged SFT model: %s | dtype=%s", model_path, dtype)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        device_map="auto",
    )

    logger.info("Loading GRPO adapter: %s", ADAPTER_DIR)
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

    logger.info("GRPO model loaded")
    model_label = f"{model_path} + GRPO LoRA adapter {ADAPTER_DIR}"
    return model, tokenizer, model_label


def main():
    args = argparse.Namespace(
        stage="GRPO",
        model_path=MODEL_DIR,
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
