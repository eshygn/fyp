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
    DPO Trainer with length-normalised log probabilities.
    Divides sequence log probs by completion length before computing
    implicit rewards, preventing longer sequences from dominating
    and reducing reward margin saturation.
    """

    def _compute_loss(self, model, inputs, return_outputs):
        import torch
        from trl.trainer.utils import selective_log_softmax
        from peft import get_peft_model
        from trl.trainer.dpo_trainer import disable_gradient_checkpointing, use_adapter

        mode = "train" if self.model.training else "eval"
        device = self.accelerator.device

        _non_model_keys = {"completion_mask", "ref_chosen_logps", "ref_rejected_logps"}
        model_kwargs = {k: v for k, v in inputs.items() if k not in _non_model_keys}
        model_kwargs["use_cache"] = False
        outputs = model(**model_kwargs)

        input_ids = inputs["input_ids"]
        completion_mask = inputs["completion_mask"]
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_completion_mask = completion_mask[..., 1:].contiguous()
        per_token_logps = selective_log_softmax(shift_logits, shift_labels)
        per_token_logps[shift_completion_mask == 0] = 0.0

        # LENGTH NORMALISATION: divide by completion length instead of summing
        completion_lens = shift_completion_mask.sum(dim=1).float().clamp(min=1.0)
        logps = per_token_logps.sum(dim=1) / completion_lens

        chosen_logps, rejected_logps = logps.chunk(2, dim=0)

        if self.precompute_ref_logps:
            ref_chosen_logps = inputs["ref_chosen_logps"]
            ref_rejected_logps = inputs["ref_rejected_logps"]
        else:
            with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
                if hasattr(model, "peft_config") and self.ref_model is None:
                    model_unwrapped = self.accelerator.unwrap_model(model)
                    with use_adapter(model_unwrapped, adapter_name="ref" if "ref" in model_unwrapped.peft_config else None):
                        ref_outputs = self.model(**model_kwargs)
                else:
                    ref_model = self.ref_model if self.ref_model is not None else model
                    ref_outputs = ref_model(**model_kwargs)

                ref_shift_logits = ref_outputs.logits[..., :-1, :].contiguous()
                ref_per_token_logps = selective_log_softmax(ref_shift_logits, shift_labels)
                ref_per_token_logps[shift_completion_mask == 0] = 0.0
                # Apply same length normalisation to reference
                ref_logps = ref_per_token_logps.sum(dim=1) / completion_lens
                ref_chosen_logps, ref_rejected_logps = ref_logps.chunk(2, dim=0)

        # Compute DPO loss using beta-scaled log ratios
        import torch.nn.functional as F
        pi_log_ratios = chosen_logps - rejected_logps
        ref_log_ratios = ref_chosen_logps - ref_rejected_logps
        logits = pi_log_ratios - ref_log_ratios
        loss = -F.logsigmoid(self.beta * logits).mean()

        # Compute metrics
        chosen_rewards = self.beta * (chosen_logps - ref_chosen_logps).detach()
        rejected_rewards = self.beta * (rejected_logps - ref_rejected_logps).detach()
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        metrics = {}
        prefix = "eval_" if mode == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().item()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().item()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().item()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().item()
        metrics[f"{prefix}logps/chosen"] = chosen_logps.detach().mean().item()
        metrics[f"{prefix}logps/rejected"] = rejected_logps.detach().mean().item()
        metrics[f"{prefix}logits/chosen"] = logits.detach().mean().item()
        metrics[f"{prefix}logits/rejected"] = (-logits).detach().mean().item()

        self.log(metrics)
        if return_outputs:
            return loss, metrics
        return loss


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
