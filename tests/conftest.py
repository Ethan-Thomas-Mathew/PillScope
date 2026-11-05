import pytest
from helpers import make_pill_photo


@pytest.fixture
def pill_photo():
    """A synthetic round 'pill' photo, BGR, plain background."""
    return make_pill_photo()
