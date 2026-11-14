import argparse
from pathlib import Path

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from patch_classifier.tools import CBIS_MAMMOGRAM, initialize_model


CLASS_NAMES = [
    "background",
    "malignant calcification",
    "benign calcification",
    "malignant mass",
    "benign mass",
]


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Evaluate the S patch classifier on a labeled patch CSV."
    )
    parser.add_argument(
        "--test-csv",
        required=True,
        help="CSV with 'path' and 'label' columns for held-out S patches.",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(root / "trained_models" / "best_s_patch_model.pt"),
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    test_csv = Path(args.test_csv)
    checkpoint = Path(args.checkpoint)
    if not test_csv.is_file():
        raise FileNotFoundError("Test CSV not found: {}".format(test_csv))
    if not checkpoint.is_file():
        raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CBIS_MAMMOGRAM(str(test_csv), transform=T.ToTensor())
    if len(dataset) == 0:
        raise ValueError("The test CSV contains no patches.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    model = initialize_model("resnet", num_classes=5, use_pretrained=False)
    state_dict = torch.load(str(checkpoint), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    class_correct = torch.zeros(5, dtype=torch.long)
    class_total = torch.zeros(5, dtype=torch.long)

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.long)
            if torch.any((labels < 0) | (labels >= len(CLASS_NAMES))):
                raise ValueError("Labels in the test CSV must be integers from 0 to 4.")
            predictions = model(images).argmax(dim=1)

            class_total += torch.bincount(labels.cpu(), minlength=5)
            correct_labels = labels[predictions == labels]
            class_correct += torch.bincount(correct_labels.cpu(), minlength=5)

    total = int(class_total.sum())
    correct = int(class_correct.sum())
    print("Device: {}".format(device))
    print("Test accuracy: {:.2f}% ({}/{})".format(100 * correct / total, correct, total))
    for index, name in enumerate(CLASS_NAMES):
        count = int(class_total[index])
        if count:
            accuracy = 100 * int(class_correct[index]) / count
            print("{}: {:.2f}% ({}/{})".format(
                name, accuracy, int(class_correct[index]), count
            ))
        else:
            print("{}: N/A (no test patches)".format(name))


if __name__ == "__main__":
    main()
