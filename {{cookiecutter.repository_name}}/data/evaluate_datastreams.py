from datastream import Datastream, samplers

from .datasets import datasets


def evaluate_datastreams():
    return {
        split_name: Datastream(dataset, samplers.SequentialSampler(len(dataset)))
        for split_name, dataset in datasets().items()
    }
