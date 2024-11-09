from functools import partial
import multiprocessing as mp
from typing import Callable
import pandas as pd
import numpy as np
import os

def apply(data: pd.DataFrame,
          function: Callable,
          chunk_size: int
          ) -> pd.DataFrame:
    ## THIS IS A PLACEHOLDER
    return pd.DataFrame([])

def wrapper(func, args, kwargs, batch):
    return func(*args, **kwargs, file_list=batch)

def file_func(func: Callable,
              dir: str,
              args: list = [],
              kwargs: dict = {},
              batch_size: int = 100) -> pd.DataFrame:
    batch_results = []
    files = list(os.listdir(dir))
    batches = []
    for i in range(0, len(files) - batch_size, batch_size):
        batches.append(files[i:i+batch_size])
    batches.append(files[-(len(files) % batch_size):])
    with mp.Pool() as pool:
        batch_results = pool.map(partial(wrapper, func, args, kwargs), batches)
    return pd.concat(batch_results, ignore_index=True)

if __name__ == "__main__":
    def test_fun(file_list=[]):
        return pd.DataFrame(file_list)
    result = file_func(test_fun, "./datasets/release_in_the_wild/")
    print(result)
