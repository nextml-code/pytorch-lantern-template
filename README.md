# Pytorch Lantern Template

Project template to be used together with [pytorch-lantern](https://github.com/nextml-code/pytorch-lantern).

## Usage

See [pytorch-lantern](https://github.com/nextml-code/pytorch-lantern) for instructions.

## Code structure

What each directory in the newly-initialized project contains:

### `data`

Problem formulation in a natural (human-readable) form. Datastreams
for training and evaluation.

- Splits data in train / early_stopping / compare
- Augments data
- Create informative batches (oversample, stratify, etc)
- Works with the natural format of the examples
- Contains train/compare and train/early stopping splits as `.json` files.

### `package`

Helper functions for running inference and downloading model weights if they are not packaged.

### `package/model`

Contains the model architecture as well as functions to convert the data from
human-readable to model-readable.

- Preprocessing natural data into model-interpretable data
- Model
- Predictions (model output representation with helper functions)
- Loss
- Prediction visualization

### `operations`

Operations are scripts that are run via the [guild CLI](https://guild.ai/)
that saves artefacts from the run such as model checkpoints.

These often include:

- `guild run train`
- `guild run retrain model=xyz`
- `guild run evaluate model=xyz`

## Development

Install dev dependencies like cookiecutter.

```bash
poetry install
```

Setup test environment

```bash
./test/create.sh
```

Download and prepare dataset

```bash
./test/prepare.sh
```

Run tests in any order

```bash
./test/run.sh
./test/test.sh
```

It can be useful to chain create and run when developing

```bash
./test/create.sh && ./test/run.sh
```

or create and prepare:

```bash
./test/create.sh && ./test/prepare.sh
```
