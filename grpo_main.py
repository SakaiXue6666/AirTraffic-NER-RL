import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from grpo_data import build_rl_dataset
from reward import ner_reward_func
from grpo_config import (
    SFT_LORA_PATH,
    GRPO_OUTPUT_DIR,
    TRAIN_FILE,
    MAX_PROMPT_LENGTH,
    MAX_COMPLETION_LENGTH,
    PER_DEVICE_TRAIN_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    NUM_TRAIN_EPOCHS,
    LOGGING_STEPS,
    SAVE_STEPS,
    SAVE_TOTAL_LIMIT,
    NUM_GENERATIONS,
    TEMPERATURE,
    TOP_P,
    BETA,
    EPSILON,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    USE_VLLM,
    SEED,
)


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        SFT_LORA_PATH,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 直接从 SFT LoRA checkpoint 继续
    model = AutoModelForCausalLM.from_pretrained(
        SFT_LORA_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    train_dataset = build_rl_dataset(TRAIN_FILE)

    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ],
    )

    training_args = GRPOConfig(
        output_dir=GRPO_OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        bf16=torch.cuda.is_available(),
        fp16=False,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        num_generations=NUM_GENERATIONS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        beta=BETA,
        epsilon=EPSILON,
        use_vllm=USE_VLLM,
        seed=SEED,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=ner_reward_func,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(GRPO_OUTPUT_DIR)
    tokenizer.save_pretrained(GRPO_OUTPUT_DIR)


if __name__ == "__main__":
    main()