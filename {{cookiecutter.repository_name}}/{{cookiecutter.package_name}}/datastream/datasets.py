from {{cookiecutter.package_name}} import problem


def datasets():
    datasets = problem.datasets()
    datasets["train"] = datasets["train"].split(
        key_column="index",
        proportions=dict(train=0.8, early_stopping=0.2),
        stratify_column="class_name",
        filepath="{{cookiecutter.package_name}}/splits/early_stopping.json",
    )

    datasets = dict(
        train=datasets["train"]["train"],
        early_stopping=datasets["train"]["early_stopping"],
        compare=datasets["compare"],
    )

    return dict(
        **datasets,
        **{f"mini_{name}": mini_subset(dataset) for name, dataset in datasets.items()},
    )


def mini_subset(dataset):
    return dataset.subset(lambda df: df["index"].isin(df.sample(256)["index"]))
