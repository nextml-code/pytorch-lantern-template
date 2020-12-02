import numpy as np


def log_examples(logger, name, epoch, examples, predictions):
    n_examples = min(5, len(predictions))
    indices = np.random.choice(
        len(predictions),
        n_examples,
        replace=False,
    )
    logger.add_images(
        f"{name}/predictions",
        np.stack(
            [
                np.stack(
                    [np.array(predictions[index].representation(examples[index]))],
                    axis=-1,
                )
                / 255
                for index in indices
            ]
        ),
        epoch,
        dataformats="NHWC",
    )
