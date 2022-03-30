import os
from abc import abstractmethod
from argparse import ArgumentParser
from typing import Type, Iterable

from computervision.data.data_modules import CIFAR10DataModule
from computervision.data.pre_processor import PreProcessor
from computervision.lightning_modules.classifier import Classifier
from computervision.train import TrainConfig, TrainerConfig
from computervision.utils.misc_utils import get_subclasses_from_object_dict


class ExperimentGroup:
    @abstractmethod
    def __iter__(self) -> Iterable[TrainConfig]:
        pass


class Cifar10(ExperimentGroup):
    def __iter__(self) -> Iterable[TrainConfig]:
        yield TrainConfig(
            name='test',
            base_output_dir='experiments/cifar10',
            data_module=CIFAR10DataModule.Config(batch_size=2048,
                                                 pre_processor=PreProcessor.Config('cifar10_default')),
            lightning_module=Classifier.Config(
                backbone='resnet18',
                optimizer_class="torch.optim.AdamW",
                optimizer_init_params={'lr': 3e-4},
                scheduler_class="torch.optim.lr_scheduler.ReduceLROnPlateau",
                scheduler_init_params={'factor': 0.1, 'patience': 5},
                scheduler_lightning_cfg={'monitor': 'val/acc', 'mode': 'min'},
            ),
            model_checkpoint=dict(
                monitor="val/loss",
                mode="max",
                filename="epoch{epoch}__step{step}__val_loss{val/loss:.2f}",
                auto_insert_metric_name=False,
                save_top_k=5,
                verbose=True,
            ),
            early_stopping=None,
            experiment_group='test',
            seed=1,
            trainer=TrainerConfig(
                gpu=1,
                max_epochs=500,
            )
        )


def main(args=None):
    parser = ArgumentParser()
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        choices=["run", "print"],
        help="if `print`, print commands to the terminal, " "if `run`, run each job in separate tmux windows",
        required=True,
    )
    parser.add_argument("--max-experiments", type=int, help="maximum experiments to run")
    parser.add_argument(
        "-e",
        "--experiment_group",
        # choices is a list of all subclasses of ExperimentGroup
        choices=list(get_subclasses_from_object_dict(ExperimentGroup, globals()).keys()),
        help="Specify the name of an ExperimentGroup",
        required=True,
    )
    args = parser.parse_args(args)
    experiment_group_class: Type[ExperimentGroup] = globals()[args.experiment_group]
    configs_list = list(experiment_group_class())
    print(f"Processing {len(configs_list)} total experiments...")

    for config in configs_list:
        print("*" * 30 + f" {config.name}" + "*" * 30)

        # rename tmux window
        # print(f"tmux rename-window {config.name}")

        os.makedirs(config.output_dir, exist_ok=True)
        path = os.path.join(config.output_dir, "train_config.yaml")
        config.to_yaml_file(path)
        cmd = f'python -m computervision.train --config {path}'

        if args.mode == "print":
            print(cmd)
        elif args.mode == 'run':
            os.system(cmd)
        else:
            raise ValueError(f"{args.mode} is invalid")

        print("*" * 72)


if __name__ == '__main__':
    main()
