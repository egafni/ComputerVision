from dataclasses import dataclass

from vision.utils.config_utils import ConfigClassMixin


@dataclass
class PreProcessingConfig(ConfigClassMixin):
    augmentation_class: str

