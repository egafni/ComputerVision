import pytest
from dacite import DaciteError

from vision.utils.config_utils import check_required, REQUIRED


def test_required():
    with pytest.raises(DaciteError):
        # make sure we raise an exception if the REQUIRED default is set for any field
        check_required(dict(a=dict(b=dict(c=REQUIRED))))