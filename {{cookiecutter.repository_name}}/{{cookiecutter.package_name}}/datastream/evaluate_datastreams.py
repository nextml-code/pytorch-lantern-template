from datastream import Datastream

from {{cookiecutter.package_name}} import datastream


def evaluate_datastreams():
    datasets = datastream.datasets()
    return {
        split_name: Datastream(dataset)
        for split_name, dataset in dict(
            gradient=datasets['gradient'],
            early_stopping=datasets['early_stopping'],
            compare=datasets['compare'],
        ).items()
    }
