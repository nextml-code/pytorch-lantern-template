import json
from pathlib import Path


def verify_splits():
    previous_splits = {
        path.name: json.loads(path.read_text())
        for path in Path("model/{{cookiecutter.package_name}}/splits").glob("*.json")
    }

    current_splits = {
        path.name: json.loads(path.read_text())
        for path in Path("{{cookiecutter.package_name}}/splits").glob("*.json")
    }

    if len(set(previous_splits.keys()) - set(current_splits.keys())) > 0:
        print("Some split files are missing in the current splits")
        return False

    checks = [
        dict(
            previous_split_name=previous_split_name,
            current_split_name=current_split_name,
            file_name=file_name,
            overlap=len(set(previous_keys).intersection(set(current_keys))) > 0,
        )
        for file_name, previous_split in previous_splits.items()
        for previous_split_name, previous_keys in previous_split.items()
        for current_split_name, current_keys in current_splits[file_name].items()
        if previous_split_name != current_split_name
    ]

    for check in checks:
        if check["overlap"]:
            print(
                f"Some keys from previous split \"{check['previous_split_name']}\""
                f" are present in current split \"{check['current_split_name']}\""
                f" from \"{check['file_name']}\""
            )

    return not any([check["overlap"] for check in checks])
