# {{cookiecutter.package_name}}

## Development

### Installation

Install [poetry](https://github.com/python-poetry/poetry>):

```
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
```

Setup environment:

```bash
poetry install
```

### Training

```
poetry shell
guild run prepare
guild run train
guild run retrain model=<model-hash>
guild run evaluate model=<model-hash>
guild tensorboard <model-hash>
```
