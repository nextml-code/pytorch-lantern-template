import argparse
import subprocess
import os
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8888)
    args = parser.parse_args()

    working_dir = f"{os.environ['GUILD_HOME']}/runs/{os.environ['RUN_ID']}"
    command = f"jupyter notebook --port={args.port}"
    os.environ["PYTHONPATH"] += os.environ["PROJECT_DIR"] + ":"

    if "PYTHONSTARTUP" in os.environ and Path(os.environ["PYTHONSTARTUP"]).exists():
        python_startup_script = (
            Path(os.environ["PYTHONSTARTUP"]).open("r").read() + "\n"
        )
    else:
        python_startup_script = ""

    Path("change-dir.py").write_text(
        f"{python_startup_script}import os\nos.chdir('{working_dir}')"
    )

    os.environ["PYTHONSTARTUP"] = f"{os.getcwd()}/change-dir.py"

    subprocess.run(command, shell=True)
