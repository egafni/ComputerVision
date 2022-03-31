# ComputerVision
Computer Vision Pytorch Framework

## Install

Install poetry: https://python-poetry.org/docs/#installation

    git clone git@github.com:egafni/ComputerVision.git
    poetry install

    # IF you have cuda11 installed, overwrite the proper version of torch
    poetry run pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 -I --no-depspoetry run pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 -I --no-deps

## Activating VirtualEnv

    poetry shell

## Testing

    poetry run pytest

## Running Experiments

    # specify your Experiment in run_experiments.py

    $ poetry run python run_experiments.py -e ExperimentGroupName -m run

ex:
    $ poetry run python run_experiments.py -e Cifar10 -m run
    $ poetry run python run_experiments.py -e DTD -m run


## Running Experiments with GPU
## Docker (TODO)

Dockerfile to build the image

## CI/CD (TODO)

Add CI/CD.  I like to use gitlab or github workflows.  Builds the docker contrainer & runs the tests. Often I will have to test runner on the development server s
For more advanced projects, I'll have the tests running a local server so that the docker images 
are easily cached and it can have access to GPUs and production/R&D data.

## Submitting Cloud Individual Jobs (TODO)

    The submit.py script allows you to run any arbitrary command (such as training a model) in the cloud
    in the exact environment of the current repository.

    $ submit.py --machine-type ... $COMMAND 
    # ex:
    # submit.py --machine-type ... train.py --config train_config.yaml

  1) Builds and pushes the docker container
  2) Runs $COMMAND on an instance in the cloud inside the pushed Docker container
  3) Streams the output of the command to the console

