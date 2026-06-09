import numpy as np
import evaluate

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from peft import LoraConfig, TaskType, get_peft_model

from config import (
    MODEL_NAME,
    DATASET_NAME,
    OUTPUT_DIR,
    ADAPTER_DIR,
    MAX_INPUT_LENGTH,
    MAX_TARGET_LENGTH,
    TRAIN_SAMPLE_SIZE,
    EVAL_SAMPLE_SIZE,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
)


def preprocess_function(batch, tokenizer):
    """
    Convert raw dialogue-summary examples into tokenized model inputs.

    Input text:
        summarize: <dialogue>

    Target text:
        <summary>
    """

    inputs = ["summarize: " + dialogue for dialogue in batch["dialogue"]]

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
    )

    labels = tokenizer(
        text_target=batch["summary"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
    )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


def compute_metrics(eval_preds, tokenizer):
    """
    Compute ROUGE scores for generated summaries.

    Some prediction/label token IDs may contain -100.
    The tokenizer cannot decode negative token IDs, so we replace -100
    with the tokenizer pad token before decoding.
    """

    rouge = evaluate.load("rouge")

    predictions, labels = eval_preds

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    predictions = np.where(
        predictions != -100,
        predictions,
        tokenizer.pad_token_id,
    )

    labels = np.where(
        labels != -100,
        labels,
        tokenizer.pad_token_id,
    )

    decoded_predictions = tokenizer.batch_decode(
        predictions,
        skip_special_tokens=True,
    )

    decoded_labels = tokenizer.batch_decode(
        labels,
        skip_special_tokens=True,
    )

    result = rouge.compute(
        predictions=decoded_predictions,
        references=decoded_labels,
        use_stemmer=True,
    )

    return {
        "rouge1": result["rouge1"],
        "rouge2": result["rouge2"],
        "rougeL": result["rougeL"],
    }
def main():
    print("Loading dataset:", DATASET_NAME)
    dataset = load_dataset(DATASET_NAME)

    print("Preparing small CPU-friendly training and validation splits...")
    train_dataset = dataset["train"].shuffle(seed=42).select(range(TRAIN_SAMPLE_SIZE))
    eval_dataset = dataset["validation"].shuffle(seed=42).select(range(EVAL_SAMPLE_SIZE))

    print("Loading tokenizer:", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading base model:", MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    print("Applying LoRA configuration...")

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q", "v"],
        bias="none",
    )

    model = get_peft_model(model, lora_config)

    print("\nTrainable parameter summary:")
    model.print_trainable_parameters()

    print("\nTokenizing training dataset...")
    tokenized_train = train_dataset.map(
        lambda batch: preprocess_function(batch, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
    )

    print("\nTokenizing validation dataset...")
    tokenized_eval = eval_dataset.map(
        lambda batch: preprocess_function(batch, tokenizer),
        batched=True,
        remove_columns=eval_dataset.column_names,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        dataloader_pin_memory=False,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        save_total_limit=2,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer),
    )

    print("\nStarting PEFT/LoRA training...")
    trainer.train()

    print("\nSaving trained LoRA adapter...")
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)

    print(f"\nTraining complete. Adapter saved at: {ADAPTER_DIR}")


if __name__ == "__main__":
    main()
