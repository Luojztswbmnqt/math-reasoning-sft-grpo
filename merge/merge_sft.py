"""Merge the trained SFT LoRA adapter into its base model."""

import json
import logging
import sys
from pathlib import Path

import torch
from peft import PeftModel
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
ADAPTER_DIR = ROOT / "output/sft/final_adapter"
MERGED_MODEL_DIR = ROOT / "output/sft/merged_model"
EXPECTED_BASE_MODEL = "Qwen/Qwen2.5-0.5B"


def validate_adapter():
    required_files = [
        ADAPTER_DIR / "adapter_config.json",
        ADAPTER_DIR / "adapter_model.safetensors",
        ADAPTER_DIR / "tokenizer_config.json",
    ]
    missing_files = [str(path) for path in required_files if not path.exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing adapter files: {missing_files}")

    with (ADAPTER_DIR / "adapter_config.json").open("r", encoding="utf-8") as file:
        adapter_config = json.load(file)

    base_model = adapter_config.get("base_model_name_or_path")
    if base_model != EXPECTED_BASE_MODEL:
        raise ValueError(
            f"The adapter expects {base_model!r}, but EXPECTED_BASE_MODEL is "
            f"{EXPECTED_BASE_MODEL!r}."
        )

    tensor_count = 0
    with safe_open(
        ADAPTER_DIR / "adapter_model.safetensors",
        framework="pt",
        device="cpu",
    ) as weights:
        for tensor_name in weights.keys():
            tensor = weights.get_tensor(tensor_name)
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Non-finite adapter tensor: {tensor_name}")
            tensor_count += 1

    logger.info(
        "Adapter validated | base_model=%s | PEFT=%s | tensors=%d",
        base_model,
        adapter_config.get("peft_version", "unknown"),
        tensor_count,
    )
    return base_model


def main():
    base_model_name = validate_adapter()

    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    if use_bf16:
        dtype = torch.bfloat16
    elif use_cuda:
        dtype = torch.float16
    else:
        dtype = torch.float32

    logger.info("Loading tokenizer: %s", ADAPTER_DIR)
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)

    logger.info("Loading base model: %s | dtype=%s", base_model_name, dtype)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map="auto" if use_cuda else None,
        low_cpu_mem_usage=True,
    )

    logger.info("Loading SFT adapter: %s", ADAPTER_DIR)
    sft_model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_DIR,
        is_trainable=False,
    )

    logger.info("Merging SFT adapter into the base model")
    merged_model = sft_model.merge_and_unload(
        progressbar=True,
        safe_merge=True,
    )

    remaining_lora_parameters = [
        name for name, _ in merged_model.named_parameters() if "lora_" in name
    ]
    if remaining_lora_parameters:
        raise RuntimeError("LoRA parameters remain after merge")

    MERGED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Saving merged model: %s", MERGED_MODEL_DIR)
    merged_model.save_pretrained(
        MERGED_MODEL_DIR,
        safe_serialization=True,
        max_shard_size="2GB",
    )
    tokenizer.save_pretrained(MERGED_MODEL_DIR)

    merge_info = {
        "base_model": base_model_name,
        "source_adapter": str(ADAPTER_DIR),
        "merged_model": str(MERGED_MODEL_DIR),
        "dtype": str(dtype),
    }
    with (MERGED_MODEL_DIR / "merge_info.json").open("w", encoding="utf-8") as file:
        json.dump(merge_info, file, indent=2)

    logger.info("Merge complete: %s", MERGED_MODEL_DIR)


if __name__ == "__main__":
    main()
