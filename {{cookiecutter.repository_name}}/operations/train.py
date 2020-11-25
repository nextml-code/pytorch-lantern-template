import argparse
import os
import json
import random
from functools import partial
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
import torch.utils.tensorboard
import torch.nn.functional as F
import logging
import lantern
from lantern.functional import starcompose
from lantern import set_seeds, worker_init
from datastream import Datastream

from {{cookiecutter.package_name}} import datastream, architecture, metrics, log_examples


def train(config):
    set_seeds(config["seed"])
    device = torch.device("cuda" if config["use_cuda"] else "cpu")

    model = architecture.Model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    if Path("model").exists():
        print("Loading model checkpoint")
        model.load_state_dict(torch.load("model/model.pt"))
        optimizer.load_state_dict(torch.load("model/optimizer.pt"))
        lantern.set_learning_rate(optimizer, config["learning_rate"])

    gradient_data_loader = datastream.GradientDatastream().data_loader(
        batch_size=config["batch_size"],
        num_workers=config["n_workers"],
        n_batches_per_epoch=config["n_batches_per_epoch"],
        worker_init_fn=partial(worker_init, config["seed"]),
        collate_fn=tuple,
    )

    evaluate_data_loaders = {
        f"evaluate_{name}": (
            datastream.take(128).data_loader(
                batch_size=config["eval_batch_size"],
                num_workers=config["n_workers"],
                collate_fn=tuple,
            )
        )
        for name, datastream in datastream.evaluate_datastreams().items()
    }

    tensorboard_logger = torch.utils.tensorboard.SummaryWriter()
    early_stopping = lantern.EarlyStopping()
    gradient_metrics = lantern.Metrics(
        name="gradient",
        tensorboard_logger=tensorboard_logger,
        metrics=metrics.gradient_metrics(),
    )

    for epoch in lantern.Epochs(config["max_epochs"]):

        with lantern.module_train(model):
            for examples in lantern.ProgressBar(
                gradient_data_loader, metrics=gradient_metrics[["loss"]]
            ):
                predictions = model.predictions(
                    architecture.FeatureBatch.from_examples(examples)
                )
                loss = predictions.loss(examples)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                (
                    gradient_metrics.update_(
                        examples, predictions.detach(), loss.detach().cpu().numpy()
                    ).log_()
                )
        gradient_metrics.print()

        evaluate_metrics = {
            name: lantern.Metrics(
                name=name,
                tensorboard_logger=tensorboard_logger,
                metrics=metrics.evaluate_metrics(),
            )
            for name in evaluate_data_loaders.keys()
        }

        with lantern.module_eval(model), torch.no_grad():
            for name, data_loader in evaluate_data_loaders.items():
                for examples in tqdm(data_loader, desc=name, leave=False):
                    predictions = model.predictions(
                        architecture.FeatureBatch.from_examples(examples)
                    )
                    loss = predictions.loss(examples)

                    evaluate_metrics[name].update_(
                        examples, predictions.detach(), loss.detach().cpu().numpy()
                    )
                evaluate_metrics[name].log_().print()

        early_stopping = early_stopping.score(
            -evaluate_metrics["evaluate_early_stopping"]["loss"].compute()
        )
        if early_stopping.scores_since_improvement == 0:
            torch.save(model.state_dict(), "model.pt")
            torch.save(optimizer.state_dict(), "optimizer.pt")
        elif early_stopping.scores_since_improvement > 5:
            break
        early_stopping.print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--n_batches_per_epoch", default=50, type=int)
    parser.add_argument("--n_batches_per_step", default=1, type=int)
    parser.add_argument("--patience", type=float, default=10)
    parser.add_argument("--n_workers", default=2, type=int)

    try:
        __IPYTHON__
        args = parser.parse_known_args()[0]
    except NameError:
        args = parser.parse_args()

    config = vars(args)
    config.update(
        seed=1,
        use_cuda=torch.cuda.is_available(),
        run_id=os.getenv("RUN_ID"),
    )

    Path("config.json").write_text(json.dumps(config))

    train(config)
