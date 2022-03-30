import logging
import subprocess
from inspect import isclass
from multiprocessing import cpu_count
from subprocess import check_output
from typing import Tuple, Type, Dict

logger = logging.getLogger(__name__)


class ClassPropertyDescriptor:
    """copied from https://stackoverflow.com/questions/5189699/how-to-make-a-class-property"""

    def __init__(self, fget, fset=None):
        self.fget = fget
        self.fset = fset

    def __get__(self, obj, klass=None):
        if klass is None:
            klass = type(obj)
        return self.fget.__get__(obj, klass)()

    def __set__(self, obj, value):
        if not self.fset:
            raise AttributeError("can't set attribute")
        type_ = type(obj)
        return self.fset.__get__(obj, type_)(value)

    def setter(self, func):
        if not isinstance(func, (classmethod, staticmethod)):
            func = classmethod(func)
        self.fset = func
        return self


def classproperty(func):
    """
    copied from https://stackoverflow.com/questions/5189699/how-to-make-a-class-property
    similar to @property decorator but works on an uninstantiated object"""
    if not isinstance(func, (classmethod, staticmethod)):
        func = classmethod(func)

    return ClassPropertyDescriptor(func)


def log_system_info():
    # log system info
    logger.info('*** System Info ****')
    logger.info(check_output("free -h", shell=True))
    logger.info(check_output("df -h", shell=True))
    logger.info(check_output("env", shell=True))
    try:
        logger.info(check_output("nvidia-smi", shell=True))
    except subprocess.SubprocessError:
        pass

    logger.info("**** CPUs ****")
    logger.info(f"{cpu_count()} CPUs detected")


def get_module_and_class_names(class_path: str) -> Tuple[str, str]:
    """
    Return module name and class name from full class path
    >>> get_module_and_class_names("torch.optim.Adam")
    ('torch.optim', 'Adam')
    """
    split = class_path.split(".")
    class_name = split[-1]
    module_name = ".".join(split[:-1])
    return module_name, class_name


def get_subclasses_from_object_dict(class_: Type, object_dict: dict) -> Dict[str, Type]:
    """
    Returns all of the subclasses of class_ that are inside object_dict, which is usually passed in as the
    globals() of the caller.

    Useful for getting all of the subclasses of a class that are defined in a module.

    :param class_: A class
    :param object_dict: an object dictionary (for ex the output of globals())
    :return: dict of {"class_name": class, ...} where class is as subtype of class_
    """
    return {
        var_name: variable
        for var_name, variable in object_dict.items()
        if isclass(variable) and issubclass(variable, class_) and var_name != class_.__name__
    }
