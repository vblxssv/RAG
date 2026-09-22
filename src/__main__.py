from src.cli import CLI


def main() -> None:
    CLI.run()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
    except KeyboardInterrupt:
        print("Interrupted")
