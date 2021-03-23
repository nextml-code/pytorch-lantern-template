import shutil
import argparse
import subprocess
import webbrowser
from enum import Enum
from pathlib import Path
from notebook.services.contents.filemanager import FileContentsManager as FCM


def jupyter(notebook, port, show):
    original_path = Path("notebooks") / args.notebook

    copy_path = Path(args.notebook)
    if original_path.exists():
        shutil.copy(original_path, copy_path)
    else:
        FCM().new(path=copy_path)

    if args.show == ShowOption.interactive:
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
    html_path = copy_path.parent / args.notebook.replace(".ipynb", ".html")
    if args.show == ShowOption.done:
        webbrowser.open(html_path)


class ShowOption(Enum):
    interactive = "interactive"
    done = "done"
    nothing = "nothing"

    def __str__(self):
        return self.value

    def __eq__(self, other):
        return str(self) == str(other)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", type=str, required=True)
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument(
        "--show",
        # cannot use type=ShowOption because choices list consists of strings
        default=str(ShowOption.interactive),
        choices=[str(option) for option in ShowOption],
        help="What to show: an interactive notebook, an HTML export when done, or nothing.",
    )
    args = parser.parse_args()
    jupyter(notebook=args.notebook, port=args.port, show=args.show)
