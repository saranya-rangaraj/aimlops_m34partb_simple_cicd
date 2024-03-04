import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd

from iris_model import __version__ as _version
from iris_model.config.core import config
from iris_model.processing.data_manager import load_pipeline
from iris_model.processing.validation import validate_inputs


pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
iris_val_pipe = load_pipeline(file_name=pipeline_file_name)

def get_species_from_index(index:int) -> str:
    for key, value in config.model_config.species_mappings.items():
        if index == value:
            return key
        

def make_prediction(*,input_data:Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """
    
    validated_data, errors = validate_inputs(input_df=pd.DataFrame(input_data))

    if not errors:
        validated_data = validated_data.reindex(columns=config.model_config.features)
        
        predictions = iris_val_pipe.predict(validated_data)
        species_pred = [get_species_from_index(index) for index in predictions.tolist()]
        results = {"predictions": species_pred,"version": _version, "errors": errors}
        print("successfully completed the prediction: ", species_pred)
    else:
        results = {"predictions": None, "version": _version, "errors": errors}
        print("prediction failed with error: ", results)

    return results

if __name__ == "__main__":
    data_in = {'petal_length':[1.2], 'petal_width':[0.4], 'sepal_length':[4.0], 'sepal_width':[2.89]}
    make_prediction(input_data=data_in)
