from dataclasses import dataclass

from computervision.utils.config_utils import ConfigClassMixin


@dataclass
class PreProcessingConfig(ConfigClassMixin):
    augmentation_class: str

