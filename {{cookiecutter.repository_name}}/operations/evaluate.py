import argparse
import os
from pathlib import Path
import json
from tqdm import tqdm
import torch
import torch.utils.tensorboard
import lantern

from {{cookiecutter.package_name}} import datastream, architecture, metrics


def evaluate(config):
    device = torch.device("cuda" if config["use_cuda"] else "cpu")

    model = architecture.Model().to(device)

    if Path("model").exists():
        print("Loading model checkpoint")
        model.load_state_dict(torch.load("model/model.pt"))

    evaluate_data_loaders = {
        f"evaluate_{name}": (
            datastream.map(
                lambda example: (
                    example,
                    architecture.StandardizedImage.from_example(example),
                )
            ).data_loader(
                batch_size=config["eval_batch_size"],
                collate_fn=lambda batch: list(zip(*batch)),
                num_workers=config["n_workers"],
            )
        )
        for name, datastream in datastream.evaluate_datastreams().items()
    }

    tensorboard_logger = torch.utils.tensorboard.SummaryWriter()
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
            for examples, features in tqdm(data_loader, desc=name, leave=False):
                predictions = model.predictions(features)
                loss = predictions.loss(examples)
                evaluate_metrics[name].update_(examples, predictions.cpu(), loss.cpu())
            evaluate_metrics[name].log_().print()

    tensorboard_logger.close()


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
