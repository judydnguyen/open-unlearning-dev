"""
SFT cold-start for RewardUnlearn self-check block.

Generates <think>+<answer> refusal examples for the forget split and runs
2 epochs of SFT so the model already knows the format before GRPO starts.

Usage:
    python scripts/sft_coldstart.py \
        --model_path open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
        --forget_split forget01 \
        --output_dir saves/sft/tofu_Llama-3.2-1B-Instruct_forget01_coldstart \
        [--num_epochs 2] [--lr 1e-5] [--batch_size 4]
"""

import argparse
import logging
import os

import torch
from datasets import Dataset, load_dataset as hf_load
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rotating refusal templates
# ---------------------------------------------------------------------------

THINK_TEMPLATES = [
    "I should check whether I have reliable information about this.\n"
    "I don't have specific information about this topic in my knowledge.",

    "Let me consider what I actually know here.\n"
    "I'm not confident I have accurate information about this.",

    "I need to be careful.\n"
    "I can't recall reliable details about this.",

    "Let me think about whether I know this.\n"
    "I don't have enough information to answer this reliably.",

    "I should be honest about the limits of my knowledge.\n"
    "I'm not sure about the specifics here.",
]

ANSWER_TEMPLATES = [
    "I don't have enough information to answer that reliably.",
    "I'm not able to recall information about that.",
    "I don't have reliable information about this topic.",
    "I'm not certain about that — I don't have specific information.",
    "I can't recall the details needed to answer this accurately.",
]

SYSTEM_PROMPT = (
    "Before answering any question, reason through what you know using "
    "<think>...</think> tags. "
    "If you are uncertain or lack specific information, express that explicitly "
    "inside the <think> block. "
    "Then provide your final answer inside <answer>...</answer> tags.\n"
    "Always follow this format:\n"
    "<think>\n[your reasoning here]\n</think>\n"
    "<answer>\n[your answer here]\n</answer>"
)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def build_sft_dataset(forget_split: str, tokenizer) -> Dataset:
    raw = hf_load("locuslab/TOFU", name=forget_split, split="train")
    logger.info(f"Loaded forget corpus: {len(raw)} samples ({forget_split})")

    records = []
    for i, sample in enumerate(raw):
        think = THINK_TEMPLATES[i % len(THINK_TEMPLATES)]
        answer = ANSWER_TEMPLATES[i % len(ANSWER_TEMPLATES)]
        assistant_text = (
            f"<think>\n{think}\n</think>\n"
            f"<answer>\n{answer}\n</answer>"
        )

        messages = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": sample["question"]},
            {"role": "assistant", "content": assistant_text},
        ]

        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        # Prompt ends where assistant turn starts — use it to mask prompt tokens.
        prompt_messages = messages[:-1]  # system + user only
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        records.append({"full_text": full_text, "prompt_text": prompt_text})

    return Dataset.from_list(records)


def tokenize_and_mask(examples, tokenizer, max_length: int = 512):
    """
    Tokenize full_text; set labels=-100 for prompt tokens so loss only
    flows through the assistant's <think>+<answer> response.
    """
    full_enc = tokenizer(
        examples["full_text"],
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    prompt_enc = tokenizer(
        examples["prompt_text"],
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    labels_batch = []
    for full_ids, prompt_ids in zip(full_enc["input_ids"], prompt_enc["input_ids"]):
        prompt_len = len(prompt_ids)
        labels = [-100] * prompt_len + full_ids[prompt_len:]
        labels_batch.append(labels)

    full_enc["labels"] = labels_batch
    return full_enc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--forget_split", default="forget01")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()

    logger.info(f"Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info("Building SFT dataset...")
    dataset = build_sft_dataset(args.forget_split, tokenizer)
    dataset = dataset.map(
        lambda ex: tokenize_and_mask(ex, tokenizer, args.max_length),
        batched=True,
        remove_columns=["full_text", "prompt_text"],
    )
    logger.info(f"SFT dataset: {len(dataset)} examples")

    logger.info(f"Loading model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=args.lr,
        warmup_steps=20,
        logging_steps=5,
        save_strategy="epoch",
        bf16=True,
        report_to="none",
        dataloader_drop_last=False,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding=True, pad_to_multiple_of=8
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    logger.info("Starting SFT cold-start...")
    trainer.train()

    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"SFT checkpoint saved to {final_dir}")

    # Quick sanity check
    logger.info("Sanity check — generating sample response...")
    model.eval()
    test_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": "What is the most well-known book written by Ayaan Costas?"},
    ]
    prompt = tokenizer.apply_chat_template(
        test_messages, tokenize=False, add_generation_prompt=True
    )
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=150, do_sample=False)
    decoded = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    logger.info(f"Sample output:\n{decoded}")


if __name__ == "__main__":
    main()
