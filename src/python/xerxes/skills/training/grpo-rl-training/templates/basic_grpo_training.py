# Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Basic grpo training module for Xerxes.

Exports:
    - MODEL_NAME
    - OUTPUT_DIR
    - MAX_PROMPT_LENGTH
    - MAX_COMPLETION_LENGTH
    - SYSTEM_PROMPT
    - get_dataset
    - extract_xml_tag
    - extract_answer
    - correctness_reward_func
    - format_reward_func
    - ... and 4 more."""

import re

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "outputs/grpo-model"
MAX_PROMPT_LENGTH = 256
MAX_COMPLETION_LENGTH = 512

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
[Your step-by-step thinking]
</reasoning>
<answer>
[Final answer]
</answer>
"""


def get_dataset(split="train"):
    """Retrieve the dataset.

    Args:
        split (Any, optional): IN: split. Defaults to 'train'. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""

    data = load_dataset("openai/gsm8k", "main")[split]

    def process_example(x):
        """Process example.

        Args:
            x (Any): IN: x. OUT: Consumed during execution.
        Returns:
            Any: OUT: Result of the operation."""

        answer = x["answer"].split("####")[1].strip() if "####" in x["answer"] else None

        return {
            "prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": x["question"]}],
            "answer": answer,
        }

    return data.map(process_example)


def extract_xml_tag(text: str, tag: str) -> str:
    """Extract xml tag.

    Args:
        text (str): IN: text. OUT: Consumed during execution.
        tag (str): IN: tag. OUT: Consumed during execution.
    Returns:
        str: OUT: Result of the operation."""

    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""


def extract_answer(text: str) -> str:
    """Extract answer.

    Args:
        text (str): IN: text. OUT: Consumed during execution.
    Returns:
        str: OUT: Result of the operation."""

    return extract_xml_tag(text, "answer")


def correctness_reward_func(prompts, completions, answer, **kwargs):
    """Correctness reward func.

    Args:
        prompts (Any): IN: prompts. OUT: Consumed during execution.
        completions (Any): IN: completions. OUT: Consumed during execution.
        answer (Any): IN: answer. OUT: Consumed during execution.
        **kwargs: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
    Returns:
        Any: OUT: Result of the operation."""

    responses = [comp[0]["content"] for comp in completions]
    extracted = [extract_answer(r) for r in responses]
    return [2.0 if ans == gt else 0.0 for ans, gt in zip(extracted, answer, strict=False)]


def format_reward_func(completions, **kwargs):
    """Format reward func.

    Args:
        completions (Any): IN: completions. OUT: Consumed during execution.
        **kwargs: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
    Returns:
        Any: OUT: Result of the operation."""

    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [comp[0]["content"] for comp in completions]
    return [0.5 if re.search(pattern, r, re.DOTALL) else 0.0 for r in responses]


def incremental_format_reward_func(completions, **kwargs):
    """Incremental format reward func.

    Args:
        completions (Any): IN: completions. OUT: Consumed during execution.
        **kwargs: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
    Returns:
        Any: OUT: Result of the operation."""

    responses = [comp[0]["content"] for comp in completions]
    rewards = []

    for r in responses:
        score = 0.0
        if "<reasoning>" in r:
            score += 0.125
        if "</reasoning>" in r:
            score += 0.125
        if "<answer>" in r:
            score += 0.125
        if "</answer>" in r:
            score += 0.125

        if "</answer>" in r:
            extra = r.split("</answer>")[-1].strip()
            score -= len(extra) * 0.001

        rewards.append(score)

    return rewards


def setup_model_and_tokenizer():
    """Setup model and tokenizer.

    Returns:
        Any: OUT: Result of the operation."""

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def get_peft_config():
    """Retrieve the peft config.

    Returns:
        Any: OUT: Result of the operation."""

    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
    )


def main():
    """Main.

    Returns:
        Any: OUT: Result of the operation."""

    print("Loading dataset...")
    dataset = get_dataset()
    print(f"Dataset size: {len(dataset)}")

    print("Loading model...")
    model, tokenizer = setup_model_and_tokenizer()

    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        run_name="grpo-training",
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=8,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        num_train_epochs=1,
        bf16=True,
        optim="adamw_8bit",
        max_grad_norm=0.1,
        logging_steps=1,
        save_steps=100,
        report_to="wandb",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            incremental_format_reward_func,
            format_reward_func,
            correctness_reward_func,
        ],
        args=training_args,
        train_dataset=dataset,
        peft_config=get_peft_config(),
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {OUTPUT_DIR}/final")
    trainer.save_model(f"{OUTPUT_DIR}/final")

    print("Training complete!")


if __name__ == "__main__":
    main()
