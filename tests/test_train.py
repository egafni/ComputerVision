import tempfile

from computervision.data.data_modules import Cifar10DataModule
from computervision.data.pre_processor import Cifar10PreProc
from computervision.lightning_modules.backbones import Resnet
from computervision.lightning_modules.classifier import Classifier
from computervision.train import TrainConfig, TrainerConfig, train_model


def test_train(tmpdir):
    # testing other architectures/datasets would entail adding another function with the right config

    tmpdir = str(tmpdir)
    config = TrainConfig(
        name='test',
        base_output_dir=tmpdir,
        data_module=Cifar10DataModule.Config(batch_size=2,
                                             num_workers=0,
                                             pre_processor=Cifar10PreProc.Config(random_crop_size=32)),
        lightning_module=Classifier.Config(
            backbone=Resnet.Config(input_size=(3, 32, 32), resnet_version='resnet18', num_classes=10, pretrained=False),
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
            accelerator='cpu',
            limit_train_batches=2,
            limit_val_batches=2,
            limit_test_batches=2,
            gpu=None,
            max_epochs=1,
        )
    )

    results, trainer, data_module = train_model(config)
    print(results)
    assert results['callback_metrics']['test/loss'] == 2.422407388687134


if __name__ == '__main__':
    # Nicer to run outside of pytest for debugging so that output is streamed to console
    # ex: CUDA_VISIBLE_DEVICES= poetry run python3 tests/test_train.py
    with tempfile.TemporaryDirectory() as tmpdir:
        test_train(tmpdir)
