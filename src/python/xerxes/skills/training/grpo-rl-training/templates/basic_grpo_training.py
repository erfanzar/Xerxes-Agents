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
"""Reference GRPO training template: GSM8K + reasoning/answer reward functions."""

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
    """Load the requested GSM8K ``split`` and map each example to the GRPO prompt format."""

    data = load_dataset("openai/gsm8k", "main")[split]

    def process_example(x):
        """Convert a GSM8K row into ``{"prompt": [...], "answer": "<final>"}``."""

        answer = x["answer"].split("####")[1].strip() if "####" in x["answer"] else None

        return {
            "prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": x["question"]}],
            "answer": answer,
        }

    return data.map(process_example)


def extract_xml_tag(text: str, tag: str) -> str:
    """Return the body of the first ``<tag>...</tag>`` block in ``text``."""

    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""


def extract_answer(text: str) -> str:
    """Return the contents of the ``<answer>`` block in ``text``."""

    return extract_xml_tag(text, "answer")


def correctness_reward_func(prompts, completions, answer, **kwargs):
    """Return 2.0 per completion whose extracted answer equals the ground truth, else 0.0."""

    responses = [comp[0]["content"] for comp in completions]
    extracted = [extract_answer(r) for r in responses]
    return [2.0 if ans == gt else 0.0 for ans, gt in zip(extracted, answer, strict=False)]


def format_reward_func(completions, **kwargs):
    """Return 0.5 when a completion contains both ``<reasoning>`` and ``<answer>``, else 0.0."""

    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [comp[0]["content"] for comp in completions]
    return [0.5 if re.search(pattern, r, re.DOTALL) else 0.0 for r in responses]


def incremental_format_reward_func(completions, **kwargs):
    """Reward per opening/closing tag present and penalise trailing junk after ``</answer>``."""

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
    """Load ``MODEL_NAME`` in bf16 with flash-attention and return ``(model, tokenizer)``."""

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def get_peft_config():
    """Return the LoRA configuration used to fine-tune the base model."""

    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
    )


def main():
    """Run the GRPO training loop end-to-end and save the final model."""

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
