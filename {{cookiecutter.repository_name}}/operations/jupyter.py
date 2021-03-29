import shutil
import argparse
import subprocess
import webbrowser
from pathlib import Path
from notebook.services.contents.filemanager import FileContentsManager


def jupyter(notebook_name, port, clean, interactive):
    original_path = Path("notebooks") / notebook_name
    notebook_path = Path(notebook_name) if clean else original_path
    if original_path.exists() and not notebook_path.exists():
        shutil.copy(original_path, notebook_path)
    if not notebook_path.exists():
        FileContentsManager().new(path=notebook_path.as_posix())

    if not clean:
        package_name = "{{cookiecutter.package_name}}"
        Path(f"sourcecode-symlink/{package_name}").rename(package_name)

    if interactive:
        subprocess.run(["jupyter", "notebook", notebook_path.as_posix()])
    else:
        subprocess.run(
            [
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--inplace",
                "--execute",
                notebook_path.as_posix(),
            ]
        )

    subprocess.run(["jupyter", "nbconvert", "--to", "html", notebook_path.as_posix()])
    if not interactive:
        webbrowser.open(notebook_path.with_suffix(".html").as_posix())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", type=str, required=True)
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument(
        "--clean",
        type=bool,
        default=False,
        help="Whether to copy notebooks and sourcecode - more inspectability, slower iteration",
    )
    parser.add_argument("--interactive", type=bool, default=True)
    args = parser.parse_args()
    jupyter(
        notebook_name=args.notebook,
        port=args.port,
        clean=args.clean,
        interactive=args.interactive,
    )
