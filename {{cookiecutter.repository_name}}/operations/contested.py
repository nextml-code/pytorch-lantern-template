import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import lantern
import torch
import torch.utils.tensorboard
from data import datasets
from datastream import Datastream, samplers
from streamlit import cli as stcli
from tqdm import tqdm
from {{cookiecutter.package_name}}.model import Model

from . import utilities


def evaluate(config):
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if config["cuda"] else "cpu")

    model = Model().to(device)

    print("Loading model checkpoint")
    model.load_state_dict(torch.load("model/model.pt"))
    data = datasets()

    datastreams = {
        f"{split_name}": Datastream(dataset, samplers.SequentialSampler(len(dataset)))
        for split_name, dataset in data.items()
    }
    evaluate_data_loaders = {
        name: (
            datastreams[name].data_loader(
                batch_size=config["eval_batch_size"],
                collate_fn=list,
                num_workers=config["n_workers"],
            )
        )
        for name, stream in datastreams.items()
    }
    losses = defaultdict(list)

    for dataset_name, data_loader in evaluate_data_loaders.items():
        print(f"Computing loss for dataset: {dataset_name}")
        for examples in lantern.ProgressBar(data_loader, dataset_name):
            with lantern.module_eval(model):
                loss = (
                    model.predictions([example.image for example in examples])
                    .loss(examples, reduction="none")
                    .cpu()
                    .numpy()
                    .tolist()
                )
                losses[dataset_name].extend(loss)

    dataframes = {
        dataset_name: dataset.dataframe.reset_index()
        .assign(
            dataset_index=range(len(losses[dataset_name])), loss=losses[dataset_name]
        )
        .sort_values(by=["loss"], ascending=False)
        for dataset_name, dataset in data.items()
    }

    path = Path("contested")
    path.mkdir()
    for dataset_name, dataframe in dataframes.items():
        dataframe[["image_path", "loss", "dataset_index", "class_name"]].to_csv(
            path / f"{dataset_name}.csv", index=False
        )


if __name__ == "__main__":
    print("Preparing datasets")
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--n_workers", type=int, default=2)
    parser.add_argument("--cuda", type=bool, choices=[True, False], default=True)
    args = parser.parse_args()

    config = vars(args)
    config.update(
        run_id=os.getenv("RUN_ID"),
        **lantern.git_info(),
    )

    Path("config.json").write_text(json.dumps(config))

    evaluate(config)

    print("Running streamlit app")
    sys.argv = ["streamlit", "run", "contested_display.py"]
    sys.exit(stcli.main())
