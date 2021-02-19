from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import numpy as np
from datastream import Dataset

from {{cookiecutter.package_name}} import problem


def CifarDataset(dataframe):
    return (
        Dataset.from_dataframe(dataframe)
        .map(
            lambda row: (
                Path(row["image_path"]),
                row["class_name"],
            )
        )
        .starmap(
            lambda image_path, class_name: problem.Example(
                image=np.array(Image.open("prepare" / image_path)),
                class_name=class_name,
            )
        )
    )


def datasets():
    # TODO: consider showing the standard case where there is no
    # predetermined split
    train_df = pd.read_csv("prepare" / problem.settings.TRAIN_CSV)
    test_df = pd.read_csv("prepare" / problem.settings.TEST_CSV)

    return dict(
        train=CifarDataset(train_df),
        compare=CifarDataset(test_df),
    )
