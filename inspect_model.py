from transformers import AutoModelForSeq2SeqLM
from config import MODEL_NAME


def main():
    print("Loading model:", MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    print("\nSearching for attention projection modules...\n")

    matched_modules = []

    for name, module in model.named_modules():
        if name.endswith(".q") or name.endswith(".v") or name.endswith(".k") or name.endswith(".o"):
            matched_modules.append(name)

    for module_name in matched_modules[:40]:
        print(module_name)

    print(f"\nTotal matched modules: {len(matched_modules)}")

    print("\nIf you see module names ending with .q and .v, then target_modules=['q', 'v'] is valid.")


if __name__ == "__main__":
    main()
