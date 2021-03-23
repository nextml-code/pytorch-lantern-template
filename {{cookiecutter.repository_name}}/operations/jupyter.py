import shutil
import argparse
import subprocess
import webbrowser
from pathlib import Path
from notebook.services.contents.filemanager import FileContentsManager as FCM


def jupyter(notebook, port, show):
    original_path = Path("notebooks") / notebook

    copy_path = Path(notebook)
    if original_path.exists():
        shutil.copy(original_path, copy_path)
    else:
        FCM().new(path=copy_path.as_posix())

    if show == show_options["interactive"]:
        subprocess.run(["jupyter", "notebook", copy_path.as_posix()])
    else:
        subprocess.run(
            [
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--inplace",
                "--execute",
                copy_path.as_posix(),
            ]
        )

    subprocess.run(["jupyter", "nbconvert", "--to", "html", copy_path.as_posix()])
    html_path = copy_path.parent / notebook.replace(".ipynb", ".html")
    if show == show_options["done"]:
        webbrowser.open(html_path)


# enums are hopeless
show_options = {option: option for option in ["interactive", "done", "nothing"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", type=str, required=True)
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument(
        "--show",
        default=show_options["interactive"],
        choices=list(show_options),
        help="What to show: an interactive notebook, an HTML export when done, or nothing.",
    )
    args = parser.parse_args()
    jupyter(notebook=args.notebook, port=args.port, show=args.show)
