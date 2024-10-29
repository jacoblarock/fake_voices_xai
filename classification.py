from typing import Tuple, Union
import numpy as np
import pandas as pd
import feature_extraction
import networks

def get_labels(path: str,
               name_col: str,
               label_col: str,
               label_0_val: str,
               label_1_val: str
               ) -> pd.DataFrame:
    data = pd.read_csv(path)
    data = data.rename({label_col: "label"})
    data = data.rename({name_col: "name"})
    data["label"] = data["label"].replace(label_0_val, 0)
    data["label"] = data["label"].replace(label_1_val, 1)
    data = data.loc[:, ["name", "label"]]
    return data

def match_labels(labels: pd.DataFrame,
                 extracted_features: list[Tuple[Union[np.ndarray, float, dict]]],
                 feature_name: str
                 ) -> pd.DataFrame:
    out = labels
    out[feature_name]
    for result in extracted_features:
        file = result[0]
    return out


def classify(matched_labels: pd.DataFrame,
             model: networks.models.Sequential
             ) -> None:
    model.fit()
