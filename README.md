# Math Reasoning Post-Training with SFT and GRPO

This project is a small-scale post-training experiment for mathematical reasoning using a two-stage pipeline:

**Supervised Fine-Tuning (SFT) → Group Relative Policy Optimization (GRPO)**

The project explores how a small language model can be improved on mathematical reasoning tasks through supervised fine-tuning followed by reinforcement learning with rule-based rewards.

## Model and Task

- Base model: `Qwen/Qwen2.5-0.5B`
- Dataset: GSM8K
- Task: Mathematical reasoning
- SFT method: LoRA
- RL algorithm: GRPO
- Reward: Rule-based final-answer correctness

## Training Pipeline

1. Prepare GSM8K training, validation, and test splits.
2. Fine-tune the base model using LoRA-based supervised fine-tuning.
3. Merge the SFT LoRA adapter into the base model.
4. Use the merged SFT model as the initialization for GRPO training.
5. Evaluate the Base, SFT, and GRPO models using the same GSM8K evaluation pipeline.

## Preliminary Results

The project follows a two-stage post-training pipeline:

**Base Model → SFT → GRPO**

Initial evaluations show progressive improvement in mathematical reasoning performance:

| Stage | GSM8K Test Accuracy | Evaluation Setting |
|---|---:|---|
| Qwen2.5-0.5B Base | 33.43% | 256-token generation limit |
| + LoRA SFT | 35.48% | 256-token generation limit |
| + GRPO (v2) | **39.27%** | 256-token generation limit |

The SFT stage improved both answer formatting and task accuracy, while GRPO further improved mathematical reasoning performance using rule-based correctness rewards.

Because the earlier Base and SFT evaluations used a shorter generation limit, these numbers should be treated as preliminary rather than a strictly controlled comparison. A final comparison under identical generation settings is being conducted.

## Current Experiments

The current implementation includes:

- LoRA-based supervised fine-tuning
- Completion-only loss for SFT
- Rule-based correctness reward for mathematical answers
- GRPO training with multiple generations per prompt
- KL regularization against the SFT reference policy
- Training metric logging and visualization
- Evaluation on the GSM8K test split

## Future Work

Planned experiments include:

- increasing the GRPO group size,
- simplifying the reward function to correctness-only reward,
- tuning KL regularization,
- analyzing group-level reward variance,
- improving reward visualization using moving-average curves,
- comparing Base, SFT, and GRPO checkpoints under identical evaluation settings.

## Project Structure

```text
math-reasoning-sft-grpo/
├── prepare_dataset.py
├── train/
│   ├── train_lora.py
│   └── train_grpo.py
├── evaluation/
│   ├── base_eval.py
│   ├── sft_eval.py
│   └── grpo_eval.py
├── merge/
│   └── merge_sft.py
├── image/
│   ├── sft_loss.png
│   └── grpo_training.jpg
└── .gitignore
