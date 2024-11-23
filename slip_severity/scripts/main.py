import argparse
from train import train
# from eval import eval 

def main():
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Train or Evaluate the LSDS model.")
    parser.add_argument("--train", action="store_true", help="Run the training process.")
    parser.add_argument("--eval", action="store_true", help="Run the evaluation process.")

    # Parse arguments
    args = parser.parse_args()

    # Execute based on arguments
    if args.train:
        print("Starting training...")
        train()

    # Develop this later
    if args.eval:
        print("Starting evaluation...")
        raise NotImplementedError

if __name__ == "__main__":
    main()
