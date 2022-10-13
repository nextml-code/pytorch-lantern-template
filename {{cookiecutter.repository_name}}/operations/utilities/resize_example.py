from {{cookiecutter.repository_name}}.model.resize import resizer


def resize_example(example):
    return example.augment(resizer)
