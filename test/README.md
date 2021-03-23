# Testing the template

In case you were as confused as me as to how the template should be tested, here's some instructions:

- Before running any scripts from this directory, make sure you have `poetry` and `cookiecutter` globally installed
- Run all scripts in this directory from the one above, i.e. the root directory of the `pytorch-lantern-template` repo
- Execute the scripts rather than sourcing them
- Run `test/create.sh` first to initialize a fake repo from the template - it'll end up under `.test-template`
- Then run all the other scripts in any order you like (except for `test/env.sh` - that one's just a helper)
