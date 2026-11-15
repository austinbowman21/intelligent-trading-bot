
import ast
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKUP_ROOT = PROJECT_ROOT / "maintenance_backups"


def check_python_file(path):
    try:
        ast.parse(path.read_text(encoding="utf-8"))
        print(f"OK: {path.relative_to(PROJECT_ROOT)}")
        return True
    except (SyntaxError, UnicodeDecodeError) as error:
        print(f"ERROR: {path}: {error}")
        return False


def check_python_files():
    errors = []

    for path in PROJECT_ROOT.rglob("*.py"):
        if ".venv" in path.parts or ".git" in path.parts:
            continue
        if not check_python_file(path):
            errors.append(path)

    print(f"\nChecked Python files. Errors: {len(errors)}")
    return errors


def backup_file(relative_path):
    source = (PROJECT_ROOT / relative_path).resolve()

    if not source.is_relative_to(PROJECT_ROOT):
        raise ValueError("File must be inside the project folder.")
    if not source.is_file():
        raise FileNotFoundError(f"File not found: {source}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination = BACKUP_ROOT / timestamp / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)

    print(f"Backup created: {destination}")
    return destination


def replace_text(relative_path, old_text, new_text):
    target = (PROJECT_ROOT / relative_path).resolve()

    if not target.is_relative_to(PROJECT_ROOT):
        raise ValueError("File must be inside the project folder.")
    if not target.is_file():
        raise FileNotFoundError(f"File not found: {target}")
    if not old_text:
        raise ValueError("The old text cannot be empty.")

    original = target.read_text(encoding="utf-8")
    matches = original.count(old_text)

    if matches != 1:
        raise ValueError(
            f"Expected exactly 1 match, but found {matches}. "
            "No changes were made."
        )

    backup_path = backup_file(relative_path)
    updated = original.replace(old_text, new_text, 1)
    target.write_text(updated, encoding="utf-8")

    if not check_python_file(target):
        shutil.copy2(backup_path, target)
        print("Syntax check failed. Original file restored.")
        return False

    print("Edit applied successfully.")
    return True


def replace_from_file(relative_path, patch_path):
    patch_path = (PROJECT_ROOT / patch_path).resolve()

    if not patch_path.is_relative_to(PROJECT_ROOT):
        raise ValueError("Patch file must be inside the project folder.")

    patch = json.loads(patch_path.read_text(encoding="utf-8"))

    if set(patch) != {"old_text", "new_text"}:
        raise ValueError("Patch must contain old_text and new_text only.")

    return replace_text(
        relative_path,
        patch["old_text"],
        patch["new_text"]
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("python tools/maintain_bot.py check")
        print("python tools/maintain_bot.py backup FILE")
        print("python tools/maintain_bot.py replace FILE OLD_TEXT NEW_TEXT")
        print("python tools/maintain_bot.py replace-file FILE PATCH_JSON")
        sys.exit(1)

    if sys.argv[1] == "check":
        sys.exit(1 if check_python_files() else 0)

    elif sys.argv[1] == "backup" and len(sys.argv) == 3:
        backup_file(sys.argv[2])

    elif sys.argv[1] == "replace" and len(sys.argv) == 5:
        success = replace_text(
            sys.argv[2], sys.argv[3], sys.argv[4]
        )
        sys.exit(0 if success else 1)

    elif sys.argv[1] == "replace-file" and len(sys.argv) == 4:
        success = replace_from_file(sys.argv[2], sys.argv[3])
        sys.exit(0 if success else 1)

    else:
        print("Invalid command or arguments.")
        sys.exit(1)
