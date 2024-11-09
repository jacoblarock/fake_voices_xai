import threading
from typing import Callable
import pandas as pd

def apply(data: pd.DataFrame,
          function: Callable,
          chunk_size: int
          ) -> pd.DataFrame:
