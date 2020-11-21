import argparse
import random
from functools import partial
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
import torch.utils.tensorboard
import torch.nn.functional as F
import logging
import wildfire
from wildfire.functional import starcompose
from wildfire import set_seeds, worker_init
from datastream import Datastream

from {{cookiecutter.package_name}} import (
    datastream, architecture, metrics, log_examples
)


def train(config):
    set_seeds(config['seed'])
    device = torch.device('cuda' if config['use_cuda'] else 'cpu')

    model = architecture.Model().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config['learning_rate']
    )

    # if Path('model').exists():
    #     print('Loading model checkpoint')
    #     wildfire.ignite.handlers.ModelCheckpoint.load(
    #         train_state, 'model/checkpoints', device
    #     )
    #     wildfire.set_learning_rate(optimizer, config['learning_rate'])

    gradient_data_loader = (
        datastream.GradientDatastream()
        .data_loader(
            batch_size=config['batch_size'],
            num_workers=config['n_workers'],
            n_batches_per_epoch=config['n_batches_per_epoch'],
            worker_init_fn=partial(worker_init, config['seed']),
            collate_fn=tuple,
        )
    )

    evaluate_data_loaders = {
        f'evaluate_{name}': (
            datastream.take(128)
            .data_loader(
                batch_size=config['eval_batch_size'],
                num_workers=config['n_workers'],
                collate_fn=tuple,
            )
        )
        for name, datastream in datastream.evaluate_datastreams().items()
    }

    tensorboard_logger = torch.utils.tensorboard.SummaryWriter()
    early_stopping = wildfire.EarlyStopping()
    gradient_metrics = wildfire.Metrics(
        name='gradient',
        tensorboard_logger=tensorboard_logger,
        metrics=dict(
            loss=wildfire.MapMetric(lambda examples, predictions, loss: loss),
        ),
    )

    for epoch in wildfire.Epochs(config['max_epochs']):

        with wildfire.module_train(model):
            for examples in wildfire.ProgressBar(
                gradient_data_loader, metrics=gradient_metrics[['loss']]
            ):
                predictions = model.predictions(
                    architecture.FeatureBatch.from_examples(examples)
                )
                loss = predictions.loss(examples)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # optional: schedule learning rate

                (
                    gradient_metrics
                    .update_(examples, predictions.detach(), loss.detach().cpu().numpy())
                    .log_()
                )
        gradient_metrics.print()

        evaluate_metrics = {
            name: wildfire.Metrics(
                name=name,
                tensorboard_logger=tensorboard_logger,
                metrics=dict(
                    loss=wildfire.MapMetric(
                        lambda examples, predictions, loss: loss
                    ),
                ),
            )
            for name in evaluate_data_loaders.keys()
        }

        with wildfire.module_eval(model), torch.no_grad():
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
            -evaluate_metrics['evaluate_early_stopping']['loss'].compute()
        )
        if early_stopping.scores_since_improvement == 0:
            torch.save(model.state_dict(), 'model.pt')
            torch.save(optimizer.state_dict(), 'optimizer.pt')
        elif early_stopping.scores_since_improvement > 5:
            break
        early_stopping.print()
