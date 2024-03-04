
"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from iris_model.config.core import config
from iris_model.processing.features import Mapper


def test_species_transformer(sample_input_data):
    # Given
    transformer = Mapper(
        variables=config.model_config.species_var, mappings=config.model_config.species_mappings
    )
    assert sample_input_data[0:3].species.tolist() == ['versicolor', 'setosa', 'virginica']

    # When
    subject = transformer.fit_transform(sample_input_data)

    # Then
    assert subject[0:3].species.tolist() == [1, 0, 2]