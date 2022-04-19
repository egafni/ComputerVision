# ComputerVision
Computer Vision Pytorch Framework.  Uses a unique method to configure complex pipelines that I've developed and iterated on over the years.
Config schemas are specified of composable dataclasses, and are specified in a python file rather than yaml.
Specifiying the config as yaml is still an option, and it is always saved for reproducability, but specifying configs in python
provides all of the flexibility of code (for loops, if/else statements, generators) to specify parameters and config compositions.

see run_experiments.py for an example.

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

examples:

    $ poetry run python scripts/run_experiments.py -e Cifar10 -m run
    # something is weird about the DTD dataset causing resizing to fail, need to fix or change datasets
    $ poetry run python scripts/run_experiments.py -e DTD -m run  


## Docker (TODO)

Dockerfile to build the image

## CI/CD (TODO)

Add CI/CD.  I like to use gitlab or github workflows.  Builds the docker contrainer & runs the tests. Often I will have to test runner on the development server s
For more advanced projects, I'll have the tests running a local server so that the docker images 
are easily cached and it can have access to GPUs and production/R&D data.

## Submitting Cloud Individual Jobs (TODO)

The scripts/submit.py script allows you to run any arbitrary command (such as training a model) in the cloud
in the exact environment of the current repository.

    $ scripts/submit.py --machine-type ... $COMMAND 
    # ex:
    # scripts/submit.py --machine-type ... train.py --config train_config.yaml

  1) Builds and pushes the docker container
  2) Runs $COMMAND on an instance in the cloud inside the pushed Docker container
  3) Streams the output of the command to the console

## Tensorboard

    $  tensorboard --logdir experiments
