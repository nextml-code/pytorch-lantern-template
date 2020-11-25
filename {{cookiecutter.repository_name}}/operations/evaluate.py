import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
from pathlib import Path
import json
from functools import partial
import numpy as np
import random
import argparse
import torch
import torch.nn.functional as F
import logging
import lantern
from lantern.functional import starcompose
from lantern import set_seeds, worker_init
from datastream import Datastream

from {{cookiecutter.package_name}} import datastream, architecture, metrics


def evaluate(config):
    device = torch.device("cuda" if config["use_cuda"] else "cpu")

    model = architecture.Model().to(device)

    # print('Loading model checkpoint')
    # lantern.ignite.handlers.ModelCheckpoint.load(
    #     train_state, 'model/checkpoints', device
    # )

    evaluate_data_loaders = {
        f"evaluate_{name}": datastream.data_loader(
            batch_size=config["eval_batch_size"],
            num_workers=config["n_workers"],
            collate_fn=tuple,
        )
        for name, datastream in datastream.evaluate_datastreams().items()
    }

    tensorboard_logger = TensorboardLogger(log_dir="tb")

    with lantern.module_eval(model), torch.no_grad():
        for name, data_loader in evaluate_data_loaders.items():

            metrics = lantern.Metrics(
                name=name,
                tensorboard_logger=tensorboard_logger,
                metrics=dict(
                    loss=lantern.MapMetric(lambda examples, predictions, loss: loss),
                ),
            )

            for examples, targets in tqdm(data_loader, desc=name, leave=False):
                predictions = model.predictions(
                    architecture.FeatureBatch.from_examples(examples)
                )
                loss = predictions.loss(examples)
                metrics[name].update_(examples, predictions, loss)
            metrics[name].log_().print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_batch_size", type=int, default=128)
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

    evaluate(config)
