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
    data = data.rename(columns={label_col: "label"})
    data = data.rename(columns={name_col: "name"})
    data["label"] = data["label"].replace(label_0_val, 0)
    data["label"] = data["label"].replace(label_1_val, 1)
    data = data.loc[:, ["name", "label"]]
    return data

def match_labels(data: pd.DataFrame,
                 extracted_features: list[Tuple[str, Union[np.ndarray, float, dict]]],
                 feature_name: str
                 ) -> pd.DataFrame:
    out = data
    out[feature_name] = None
    for result in extracted_features:
        name = result[0]
        print(name)
        if len(out[out["name"] == name]) == 0:
            index = len(out)
            plain_name = name
            while 48 <= ord(plain_name[-1]) <= 57:
                plain_name = plain_name[:-1]
            plain_index = out[out["name"] == plain_name].index[0]
            out.loc[index] = out.loc[plain_index]
        else:
            index = out[out["name"] == name].index[0]
        out.at[index, feature_name] = result[1]
    return out


def classify(matched_labels: pd.DataFrame,
             model: networks.models.Sequential
             ) -> None:
    model.fit()
