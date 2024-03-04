from typing import Any, List, Optional

from pydantic import BaseModel
from iris_model.processing.validation import DataInputSchema


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    #predictions: Optional[List[int]]
    predictions: Optional[Any]


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "petal_length" : 1.2,
                        "petal_width" : 0.4,
                        "sepal_length" : 4.0,
                        "sepal_width" : 2.89,
                    }
                ]
            }
        }
