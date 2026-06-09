from datasets import load_dataset
from config import DATASET_NAME

def main():
    print("Loading dataset:", DATASET_NAME)
    dataset = load_dataset(DATASET_NAME)

    print("\nAvailable splits:")
    print(dataset)

    print("\nTrain columns:")
    print(dataset["train"].column_names)

    print("\nFirst training example:")
    sample = dataset["train"][0]

    print("\nDialogue:")
    print(sample["dialogue"])

    print("\nSummary:")
    print(sample["summary"])

if __name__ == "__main__":
    main()
