import fire
from src.cli import CLI


def main() -> None:
    """Run the main Fire CLI application."""
    fire.Fire(CLI)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
    except KeyboardInterrupt:
        print("Interrupted")
