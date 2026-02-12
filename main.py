import argparse
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Number plate recognition baseline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("train", help="Train EMNIST digit CNN")
    subparsers.add_parser("infer", help="Run inference on plate image(s)")

    return parser.parse_args(sys.argv[1:2])


def main() -> None:
    args = parse_args()

    # Remove subcommand token so downstream parser sees only command-specific flags.
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if args.command == "train":
        from src.train_emnist import main as train_main

        train_main()
    elif args.command == "infer":
        from src.infer_plate import main as infer_main

        infer_main()


if __name__ == "__main__":
    main()
