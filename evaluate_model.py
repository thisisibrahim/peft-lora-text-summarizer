import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

from config import MODEL_NAME, DATASET_NAME, ADAPTER_DIR, MAX_INPUT_LENGTH


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


def main():
    print("Loading dataset:", DATASET_NAME)
    dataset = load_dataset(DATASET_NAME)

    test_samples = dataset["test"].select(range(3))

    print("Loading tokenizer:", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading base model...")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    base_model.eval()

    print("Loading LoRA fine-tuned model...")
    lora_base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    lora_model = PeftModel.from_pretrained(lora_base_model, ADAPTER_DIR)
    lora_model.eval()

    for index, sample in enumerate(test_samples, start=1):
        dialogue = sample["dialogue"]
        reference_summary = sample["summary"]

        base_summary = generate_summary(dialogue, base_model, tokenizer)
        lora_summary = generate_summary(dialogue, lora_model, tokenizer)

        print("\n" + "=" * 80)
        print(f"Example {index}")
        print("=" * 80)

        print("\nDialogue:")
        print(dialogue)

        print("\nReference Summary:")
        print(reference_summary)

        print("\nBase Model Summary:")
        print(base_summary)

        print("\nLoRA Fine-Tuned Summary:")
        print(lora_summary)


if __name__ == "__main__":
    main()
