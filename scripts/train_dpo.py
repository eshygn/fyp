"""
DPO Fine-tuning of Qwen3-4B for Emotional Depth in Short Stories.

Three experimental conditions:
  1. baseline  - No training, just inference with base Qwen3-4B
  2. dpo       - Standard DPO (Rafailov et al.)
  3. dpo_ln    - Length-Normalised DPO (our contribution)

Usage:
  # Standard DPO, beta=0.1
  python train_dpo.py --mode dpo --beta 0.1 --run_name dpo_b01

  # Length-normalised DPO, beta=0.1
  python train_dpo.py --mode dpo_ln --beta 0.1 --run_name dpo_ln_b01

  # Beta ablation: just change --beta
  python train_dpo.py --mode dpo_ln --beta 0.3 --run_name dpo_ln_b03
"""

import argparse
import os
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig


class LengthNormalisedDPOTrainer(DPOTrainer):
    """
    DPO Trainer with length-normalised implicit rewards.

    Standard DPO computes:
        reward(x, y) = β * [log π_θ(y|x) - log π_ref(y|x)]

    This causes training saturation because longer sequences accumulate
    more log-probability mass, dominating the reward signal
    (observed by Natarajan 2025 at ~step 20 with reward ceiling of 17).

    Length-normalised DPO instead computes:
        reward(x, y) = β * [log π_θ(y|x) - log π_ref(y|x)] / |y|

    This prevents longer sequences from dominating and allows continued
    learning beyond the saturation point.
    """

    def dpo_loss(
        self,
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
    ):
        """Override DPO loss to normalise by response length."""
        # Compute per-sequence rewards (these are summed log probs)
        chosen_rewards = policy_chosen_logps - reference_chosen_logps
        rejected_rewards = policy_rejected_logps - reference_rejected_logps

        # Normalise by stored sequence lengths
        if (hasattr(self, '_chosen_lengths') and
            hasattr(self, '_rejected_lengths') and
            self._chosen_lengths is not None and
            self._rejected_lengths is not None):
            chosen_lengths = self._chosen_lengths.clamp(min=1).float()
            rejected_lengths = self._rejected_lengths.clamp(min=1).float()
            chosen_rewards = chosen_rewards / chosen_lengths
            rejected_rewards = rejected_rewards / rejected_lengths

        # Standard sigmoid DPO loss with normalised rewards
        logits = chosen_rewards - rejected_rewards
        loss = -F.logsigmoid(self.beta * logits)

        # Metrics for logging
        chosen_reward_mean = chosen_rewards.detach().mean()
        rejected_reward_mean = rejected_rewards.detach().mean()
        reward_margin = (chosen_rewards - rejected_rewards).detach().mean()

        return loss.mean(), chosen_reward_mean, rejected_reward_mean

    def get_batch_loss_metrics(self, model, batch, train_eval="train"):
        """Override to capture response lengths before loss computation."""
        # Extract response lengths from the batch labels
        # In DPO, the batch contains concatenated chosen+rejected sequences
        # We need to figure out the response token counts

        # Get the labels - non-padding, non-prompt tokens
        if 'chosen_labels' in batch:
            chosen_labels = batch['chosen_labels']
            rejected_labels = batch['rejected_labels']
            # Count non-(-100) tokens as response length
            self._chosen_lengths = (chosen_labels != -100).sum(dim=1)
            self._rejected_lengths = (rejected_labels != -100).sum(dim=1)
        else:
            # Fallback: try to get lengths from input_ids
            self._chosen_lengths = None
            self._rejected_lengths = None

        # Call parent's method which will in turn call our overridden dpo_loss
        return super().get_batch_loss_metrics(model, batch, train_eval)


def get_model_and_tokenizer(model_name: str, use_4bit: bool = True):
    """Load Qwen3-4B with optional 4-bit quantization."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

    return model, tokenizer


def get_lora_config():
    """LoRA configuration for Qwen3-4B."""
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )


def main():
    parser = argparse.ArgumentParser(description='DPO Training for Emotional Depth')
    parser.add_argument('--mode', type=str, choices=['dpo', 'dpo_ln'],
                        required=True, help='dpo=standard, dpo_ln=length-normalised')
    parser.add_argument('--beta', type=float, default=0.1,
                        help='DPO beta parameter (controls KL penalty strength)')
    parser.add_argument('--run_name', type=str, required=True,
                        help='Name for this run (used in output dir)')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-4B',
                        help='Base model to fine-tune')
    parser.add_argument('--data_dir', type=str, default='../data/prepared',
                        help='Directory with prepared train/test datasets')
    parser.add_argument('--output_dir', type=str, default='../models',
                        help='Base output directory for saved models')
    parser.add_argument('--max_length', type=int, default=1024,
                        help='Max total sequence length (prompt + response)')
    parser.add_argument('--max_prompt_length', type=int, default=384,
                        help='Max prompt length in tokens')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Per-device batch size')
    parser.add_argument('--grad_accum', type=int, default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=1,
                        help='Number of training epochs')
    parser.add_argument('--max_steps', type=int, default=-1,
                        help='Max training steps (-1 = use epochs)')
    parser.add_argument('--logging_steps', type=int, default=10,
                        help='Log every N steps')
    parser.add_argument('--save_steps', type=int, default=100,
                        help='Save checkpoint every N steps')
    parser.add_argument('--use_4bit', action='store_true', default=True,
                        help='Use 4-bit quantization')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    # Output directory for this run
    run_output_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(run_output_dir, exist_ok=True)

    print("=" * 60)
    print(f"DPO Training - {args.mode.upper()}")
    print(f"  Beta:           {args.beta}")
    print(f"  Run name:       {args.run_name}")
    print(f"  Model:          {args.model_name}")
    print(f"  Max length:     {args.max_length}")
    print(f"  Batch size:     {args.batch_size} x {args.grad_accum} accum")
    print(f"  Learning rate:  {args.learning_rate}")
    print(f"  Output:         {run_output_dir}")
    print("=" * 60)

    # Load dataset
    print("\nLoading dataset...")
    train_ds = load_from_disk(os.path.join(args.data_dir, 'train'))
    print(f"  Train samples: {len(train_ds)}")

    # Load model and tokenizer
    print(f"\nLoading model: {args.model_name}...")
    model, tokenizer = get_model_and_tokenizer(args.model_name, args.use_4bit)

    # Apply LoRA
    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # DPO training config
    training_args = DPOConfig(
        output_dir=run_output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        beta=args.beta,
        max_length=args.max_length,
        
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=True,
        seed=args.seed,
        remove_unused_columns=False,
        run_name=args.run_name,
        report_to="none",  # change to "wandb" if you set up wandb
        gradient_checkpointing=True,
    )

    # Select trainer class based on mode
    if args.mode == 'dpo':
        print("\nUsing STANDARD DPO trainer")
        trainer_cls = DPOTrainer
    elif args.mode == 'dpo_ln':
        print("\nUsing LENGTH-NORMALISED DPO trainer")
        trainer_cls = LengthNormalisedDPOTrainer

    # Create trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )

    # Train
    print("\nStarting training...")
    train_result = trainer.train()

    # Save final model
    print(f"\nSaving model to {run_output_dir}/final...")
    trainer.save_model(os.path.join(run_output_dir, 'final'))
    tokenizer.save_pretrained(os.path.join(run_output_dir, 'final'))

    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"  Model saved to: {run_output_dir}/final")
    print(f"  Training loss:  {metrics.get('train_loss', 'N/A')}")
    print("=" * 60)


if __name__ == '__main__':
    main()
