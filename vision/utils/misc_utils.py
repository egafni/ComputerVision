import logging
import subprocess
from multiprocessing import cpu_count
from subprocess import check_output

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