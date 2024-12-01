import networks
from keras._tf_keras.keras import models, layers, Model
import pandas as pd
import numpy as np
import lime

def explain(model: Model,
            features: list[pd.DataFrame],
            feature_cols: list[str]
            ):
