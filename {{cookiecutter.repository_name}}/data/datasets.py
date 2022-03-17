from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import numpy as np
from datastream import Dataset

from .example import Example


def datasets():
    return (
        Dataset.from_dataframe(pd.read_csv(f"prepare/data.csv"))
        .map(
            lambda row: (
                Path(row["image_path"]),
                row["class_name"],
            )
        )
        .starmap(
            lambda image_path, class_name: Example(
                image=np.array(Image.open("prepare" / image_path)),
                class_name=class_name,
            )
        )
        .split(
            key_column="index",
            proportions=dict(train=0.4, early_stopping=0.2, compare=0.4),
            stratify_column="class_name",
            filepath="data/splits/early_stopping.json",
        )
    )
