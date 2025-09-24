import pytest
from unittest.mock import Mock
import numpy as np
from alpha_DocumentContextManager import DocumentContextManager

# Fixture to create a DocumentContextManager instance with mocked dependencies
@pytest.fixture
def doc_manager(mocker):
    # MOck Sentence Transformers
    mocker.patch()
