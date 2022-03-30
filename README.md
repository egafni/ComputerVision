# ComputerVision
Computer Vision Pytorch Framework

## Install

Install poetry: https://python-poetry.org/docs/#installation

    git clone git@github.com:egafni/ComputerVision.git
    poetry install

## Activating VirtualEnv

    poetry shell

## Testing

    poetry run pytest

## Training

    python train.py

## Docker (TODO)

* Dockerfile to build the image

## Submitting a Bunch of Experiments

    # specify your Experiment in run_experiments.py

    $ python run_experiments.py -e ExperimentName

## Submitting Cloud Individual Jobs

    The submit.py script allows you to run any arbitrary command (such as training a model) in the cloud
    in the exact environment of the current repository.

    $ submit.py --machine-type ... $COMMAND 
    # ex:
    # submit.py --machine-type ... train.py --config train_config.yaml

  1) Builds and pushes the docker container
  2) Runs $COMMAND on an instance in the cloud inside the pushed Docker container
  3) Streams the output of the command to the console

