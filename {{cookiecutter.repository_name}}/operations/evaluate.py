import argparse
import os
from pathlib import Path
import json
from tqdm import tqdm
import torch
import torch.utils.tensorboard
import lantern

import data
from . import utilities
from {{cookiecutter.package_name}}.model import Model


def evaluate(config):
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if config["cuda"] else "cpu")

    model = Model().to(device)

    print("Loading model checkpoint")
    model.load_state_dict(torch.load("model/model.pt"))

    datastreams = data.datastreams()
    evaluate_data_loaders = {
        name: (
            datastreams[name]
            .data_loader(
                batch_size=config["eval_batch_size"],
                collate_fn=list,
                num_workers=config["n_workers"],
            )
        )
        for name in ["evaluate_train", "evaluate_early_stopping", "evaluate_compare"]
    }

    tensorboard_logger = torch.utils.tensorboard.SummaryWriter()
    evaluate_metrics = {
        name: utilities.metrics.evaluate_metrics() for name in evaluate_data_loaders
    }

    for dataset_name, data_loader in evaluate_data_loaders.items():
        for examples in lantern.ProgressBar(
            data_loader, dataset_name
        ):
            with lantern.module_eval(model):
                predictions = model.predictions([example.image for example in examples])

            evaluate_metrics[dataset_name]["pairs"].update_(predictions, examples)

        for metric in evaluate_metrics[dataset_name].values():
            metric.log_dict(tensorboard_logger, dataset_name)

        print(lantern.MetricTable(dataset_name, evaluate_metrics[dataset_name]))

    tensorboard_logger.close()


if __name__ == "__main__":
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
