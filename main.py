import argparse
from src.train import train_pipeline
from src.predict import predict_text


def main():
    parser = argparse.ArgumentParser(description="Turkish Fake News Detector")

    # dataset path for training (file OR folder)
    parser.add_argument(
        "--train", type=str, help="Path to dataset (CSV or folder with Fake/Real)"
    )

    # text for prediction
    parser.add_argument("--predict", type=str, help="Text to classify")

    args = parser.parse_args()

    if args.train:
        train_pipeline(args.train)

    if args.predict:
        predict_text(args.predict)

    if not args.train and not args.predict:
        parser.print_help()


if __name__ == "__main__":
    main()
