import torch
import evaluate

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

from config import MODEL_NAME, DATASET_NAME, ADAPTER_DIR, MAX_INPUT_LENGTH


TEST_SAMPLE_SIZE = 30


def generate_summary(text, model, tokenizer):
    input_text = "summarize: " + text

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
    )

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=80,
            num_beams=4,
            early_stopping=True,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def compute_rouge(predictions, references):
    rouge = evaluate.load("rouge")

    return rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True,
    )


def main():
    print("Loading dataset:", DATASET_NAME)
    dataset = load_dataset(DATASET_NAME)
    test_samples = dataset["test"].select(range(TEST_SAMPLE_SIZE))

    print("Loading tokenizer:", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading base model...")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    base_model.eval()

    print("Loading LoRA model...")
    lora_base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    lora_model = PeftModel.from_pretrained(lora_base_model, ADAPTER_DIR)
    lora_model.eval()

    references = []
    base_predictions = []
    lora_predictions = []

    for index, sample in enumerate(test_samples, start=1):
        dialogue = sample["dialogue"]
        reference = sample["summary"]

        print(f"Generating summaries for sample {index}/{TEST_SAMPLE_SIZE}...")

        base_summary = generate_summary(dialogue, base_model, tokenizer)
        lora_summary = generate_summary(dialogue, lora_model, tokenizer)

        references.append(reference)
        base_predictions.append(base_summary)
        lora_predictions.append(lora_summary)

    print("\nComputing ROUGE for base model...")
    base_scores = compute_rouge(base_predictions, references)

    print("\nComputing ROUGE for LoRA model...")
    lora_scores = compute_rouge(lora_predictions, references)

    print("\n" + "=" * 80)
    print("ROUGE Comparison")
    print("=" * 80)

    print("\nBase Model:")
    print(f"ROUGE-1: {base_scores['rouge1']:.4f}")
    print(f"ROUGE-2: {base_scores['rouge2']:.4f}")
    print(f"ROUGE-L: {base_scores['rougeL']:.4f}")

    print("\nLoRA Fine-Tuned Model:")
    print(f"ROUGE-1: {lora_scores['rouge1']:.4f}")
    print(f"ROUGE-2: {lora_scores['rouge2']:.4f}")
    print(f"ROUGE-L: {lora_scores['rougeL']:.4f}")


if __name__ == "__main__":
    main()
