"""Main entrypoint for the RAG CLI application."""

import json
import sys
import fire
from pydantic import ValidationError
from src.cli import CLI


def main() -> None:
    """Run the main Fire CLI application."""
    fire.Fire(CLI)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(130)
    except FileNotFoundError as e:
        print(f"Error (File Not Found): {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error (Malformed JSON): {e}")
        sys.exit(1)
    except ValidationError as e:
        print(f"Error (Validation Error): {e.error_count()} error(s)")
        for err in e.errors()[:3]:
            loc = " -> ".join(
                str(x) for x in err["loc"] if not str(x).isdigit()
            )
            prefix = f"{loc}: " if loc else ""
            print(f"  - {prefix}{err['msg']}")
        sys.exit(1)
    except (ValueError, TypeError) as e:
        print(f"Error (Invalid Argument): {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected Error: {e}")
        sys.exit(1)
