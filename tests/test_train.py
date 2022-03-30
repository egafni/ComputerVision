import tempfile

from computervision.data.data_modules import CIFAR10DataModule
from computervision.lightning_modules.classifier import Classifier
from computervision.train import TrainConfig, TrainerConfig, train_model


def test_train(tmpdir):
    tmpdir = str(tmpdir)
    config = TrainConfig(
        name='test',
        base_output_dir=tmpdir,
        data_module=CIFAR10DataModule.Config(batch_size=2),
        lightning_module=Classifier.Config(
            backbone='resnet18'
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
            limit_train_batches=2,
            limit_val_batches=2,
            limit_test_batches=2,
            gpu=None,
            max_epochs=1,
        )
    )

    train_model(config)

if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as tmpdir:
        test_train(tmpdir)
