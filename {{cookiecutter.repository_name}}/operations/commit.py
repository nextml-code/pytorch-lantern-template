import shutil
from pathlib import Path
from tqdm.auto import tqdm
from CleanNB.clean_notebook import clean_notebook


clean = True


def commit():
    ran_dir = Path("ran-notebooks")
    target_dir = Path("notebooks")
    loading_bar = tqdm(list(ran_dir.glob("*.ipynb")))
    for ran_path in loading_bar:
        loading_bar.set_description(ran_path.name)
        target_path = target_dir / ran_path.name
        shutil.copy(ran_path, target_path)
        if clean:
            clean_notebook(target_path.as_posix(), new=False)


if __name__ == "__main__":
    commit()
