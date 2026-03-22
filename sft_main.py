import torch
from transformers import Trainer, TrainingArguments
from config import (
    TRAIN_FILE,
    OUTPUT_DIR,
    BATCH_SIZE,
    GRAD_ACCUM_STEPS,
    NUM_EPOCHS,
    LEARNING_RATE,
    LOGGING_STEPS,
    SAVE_STEPS,
)
from sft_data import build_dataset, tokenize_dataset
from model import load_tokenizer, load_model, apply_lora


def main():
    tokenizer = load_tokenizer()
    model = load_model()
    model = apply_lora(model)

    train_dataset = build_dataset(TRAIN_FILE)
    train_dataset = tokenize_dataset(train_dataset, tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()