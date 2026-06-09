import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

from config import MODEL_NAME, ADAPTER_DIR, MAX_INPUT_LENGTH


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
    print("Loading tokenizer:", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading base model:", MODEL_NAME)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    print("Loading trained LoRA adapter from:", ADAPTER_DIR)
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

    model.eval()

    sample_dialogue = '''
John: Hey Sara, are you joining the project meeting today?
Sara: I might be late. I have a doctor's appointment.
John: No problem. Should I update the team?
Sara: Yes, please tell them I will join after 30 minutes.
John: Sure, I will inform everyone.
'''

    print("\nInput Dialogue:")
    print(sample_dialogue)

    summary = generate_summary(sample_dialogue, model, tokenizer)

    print("\nLoRA Fine-Tuned Summary:")
    print(summary)


if __name__ == "__main__":
    main()
