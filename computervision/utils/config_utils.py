import dataclasses
import os
from dataclasses import fields
from typing import Any

import dacite
import fsspec
import yaml
from dacite import DaciteError

from computervision.utils.misc_utils import classproperty


class _REQUIRED:
    """
    Dataclass inheritance is horrible due to the requirement that defaults follow non-defaults.  This is a hacky way
    around that until python3.10 when we can set @dataclass(kw_only=True),
    """
    pass


REQUIRED: Any = _REQUIRED


def iter_items_dict_or_dataclass(x):
    """
    if x is a dict, iterates over key,val tuples
    if x is a dataclass, iterates over key,val tuples of its dataclass field names and values
    """
    if isinstance(x, dict):
        yield from x.items()
    if dataclasses.is_dataclass(x):
        yield from dataclasses.asdict(x).items()


def check_required(obj, path=""):
    """
    check all REQUIRED fields were set

    :param obj: either a (nested) dataclass or a (nested) dict
    """

    if isinstance(obj, dict) or dataclasses.is_dataclass(obj):
        for k, v in iter_items_dict_or_dataclass(obj):
            check_required(v, path=os.path.join(path, k))
    elif obj is REQUIRED:
        raise DaciteError(f"{path} is a required field")


class ConfigMixin:
    """
    A mixin for a dataclass used to create composable Configuration objects.
    """

    def __post_init__(self):
        check_required(self)

        for field in self.fields:
            if field.name == "unique_config_id":
                # require that unique_config_id is set to the default value so that we're instantiating
                # the correct class.  This is required by dacite's UnionType[] support, which continues trying each
                # type after a failure
                unique_config_id = getattr(self, field.name)
                if unique_config_id != field.default:
                    raise DaciteError(
                        f"unique_config_id `{unique_config_id}`" f" should be {field.default} to instantiate this class"
                    )

            if "choices" in field.metadata:
                val = getattr(self, field.name)
                if val not in field.metadata["choices"]:
                    raise ValueError(f'{field.name} is invalid, it must be in {field.metadata["choices"]}')

    @classproperty
    def fields(cls):
        return fields(cls)

    @classproperty
    def field_names(cls):
        return [f.name for f in cls.fields]

    @classmethod
    def get_name_to_field(cls):
        return dict(zip(cls.field_names, cls.fields))

    def __repr__(self):
        keys = ",".join(self.field_names)
        return f"{self.__class__.__name__}({keys})"

    def to_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data):
        # fixme we should enable strict_unions_match=True to make sure only one of the UnionType matches
        obj = dacite.from_dict(cls, data, dacite.Config(strict=True))
        check_required(obj)
        return obj

    def __iter__(self):
        for field_name in self.field_names:
            yield getattr(self, field_name)

    @classmethod
    def from_yaml_file(cls, path):
        with fsspec.open(path) as fp:
            s = fp.read().decode()

        return cls.from_yaml(s)

    @classmethod
    def from_yaml(cls, yaml_string):
        return cls.from_dict(yaml.load(yaml_string, Loader=yaml.Loader))

    def to_yaml(self):
        return yaml.dump(self.to_dict())

    def to_yaml_file(self, path):
        with fsspec.open(path, "w") as fp:
            fp.write(self.to_yaml())

    def interpolated(self):
        raise NotImplementedError('interpolated not implemented, normally this would '
                                  'allow you to set values as references to other config variables')
