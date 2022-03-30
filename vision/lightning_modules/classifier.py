from dataclasses import dataclass

import pytorch_lightning

from vision.utils.config_utils import ConfigClassMixin


class Classifier(pytorch_lightning.LightningModule):
    @dataclass
    class Config(ConfigClassMixin):
        backbone: str
        unique_config_id: str = 'ClassifierConfig'

    def __init__(self, config: Config):
        super(Classifier, self).__init__()

        self.model = config.backbone
