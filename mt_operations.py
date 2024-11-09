from functools import partial
import multiprocessing as mp
import pickle
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
              directory: str,
              cache: bool = True,
              use_cached: bool = True,
              args: list = [],
              kwargs: dict = {},
              batch_size: int = 100) -> pd.DataFrame:
    if directory[-1] != "/":
        directory = directory + "/"
    cache_path = "./cache/extracted_features/" + directory[:-1].split("/")[-1] + "_" + func.__name__
    if os.path.isfile(cache_path) and use_cached:
        with open(cache_path, "rb") as file:
            return pickle.load(file)
    batch_results = []
    files = list(os.listdir(directory))
    batches = []
    for i in range(0, len(files) - batch_size, batch_size):
        batches.append(files[i:i+batch_size])
    batches.append(files[-(len(files) % batch_size):])
    with mp.Pool() as pool:
        batch_results = pool.map(partial(wrapper, func, args, kwargs), batches)
    out = pd.concat(batch_results, ignore_index=True)
    if cache:
        with open(cache_path, "wb") as file:
            pickle.dump(pd.DataFrame(out), file)
    return out

if __name__ == "__main__":
    def test_fun(file_list=[]):
        return pd.DataFrame(file_list)
    result = file_func(test_fun, "./datasets/release_in_the_wild/")
    print(result)
