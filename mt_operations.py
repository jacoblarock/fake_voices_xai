from functools import partial
import multiprocessing as mp
import pickle
from typing import Callable
import pandas as pd
import numpy as np
import os

max_threads = 32

def apply_wrapper(data: pd.DataFrame | pd.Series,
                  function: Callable,
                  indices: list
                  ) -> pd.DataFrame | pd.Series:
    """
    MEANT FOR INTERNAL USE
    Used by the workers of the multithreaded apply function
    Applies a function to specified indices of a dataframe
    Arguments:
    - data: dataframe to which to apply
    - function: function to apply
    - indices: indices to which to apply
    """
    # data.loc[indices] = data.loc[indices].apply(function)
    if type(data) == pd.DataFrame:
        out = pd.DataFrame(columns=data.columns)
    else:
        out = pd.Series()
    for i in indices:
        out.loc[i] = function(data.loc[i])
    return out


def apply(data: pd.DataFrame | pd.Series,
          function: Callable
          ) -> pd.DataFrame | pd.Series:
    """
    A multithreaded implementation of df.apply for optimization purposes
    The number of threads used is equal to the number of available threads with a maximum of 32
    Arguments:
    - data: dataframe to which to apply
    - function: function to apply
    """
    indices = data.index
    batches = []
    batch_size = max(len(data) // min(mp.cpu_count(), max_threads), 1)
    for i in range(0, len(indices) - batch_size, batch_size):
        batches.append(list(indices)[i:i+batch_size])
    if len(indices) % batch_size != 0:
        batches.append(list(indices)[-(len(indices) % batch_size):])
    # jobs = []
    # for batch in batches:
    #     jobs.append(mp.Process(target=apply_wrapper, args=(data, function, batch)))
    #     jobs[-1].start()
    # for job in jobs:
    #     job.join()
    with mp.Pool(min(mp.cpu_count(), max_threads)) as pool:
        batch_results = pool.map(partial(apply_wrapper, data, function), batches)
    print(batch_results)
    out = pd.concat(batch_results, ignore_index=True)
    print(out)
    # with mp.Pool(min(mp.cpu_count(), max_threads)) as pool:
    #     for batch in batches:
    #         print(batch)
    #         print()
    #         pool.apply_async(apply_wrapper, (data, function, batch))
    return out

def wrapper(func, args, kwargs, batch):
    return func(*args, **kwargs, file_list=batch)

def file_func(func: Callable,
              directory: str,
              cache: bool = True,
              cache_name: str | None = None,
              use_cached: bool = True,
              args: list = [],
              kwargs: dict = {},
              batch_size: int = 100) -> pd.DataFrame:
    """
    A multithreaded implimentation to apply a function to the contents of files in a directory
    Arguments:
    - func: function to apply
    - directory: directory to search
    Keyword arguments:
    - cache: when True, results will be cached
    - use_cached: when True, previously cached results will be used
    - args: additional arguments to pass to func
    - kwargs: additional kwargs to pass to the func
    - batch_size: number of files to process in a single batch (by a single worker)
    """
    if directory[-1] != "/":
        directory = directory + "/"
    if cache_name == None:
        cache_path = "./cache/extracted_features/" + directory[:-1].split("/")[-1] + "_" + func.__name__
    else:
        cache_path = "./cache/extracted_features/" + directory[:-1].split("/")[-1] + "_" + cache_name
    if os.path.isfile(cache_path) and use_cached:
        with open(cache_path, "rb") as file:
            return pickle.load(file)
    batch_results = []
    files = list(os.listdir(directory))
    batches = []
    for i in range(0, len(files) - batch_size, batch_size):
        batches.append(files[i:i+batch_size])
    if len(files) % batch_size != 0:
        batches.append(files[-(len(files) % batch_size):])
    with mp.Pool(min(mp.cpu_count(), max_threads)) as pool:
        batch_results = pool.map(partial(wrapper, func, args, kwargs), batches)
    out = pd.concat(batch_results, ignore_index=True)
    if cache:
        with open(cache_path, "wb") as file:
            pickle.dump(pd.DataFrame(out), file)
    return out

if __name__ == "__main__":
    def test_fun(row):
        return row ** 2
    # test = pd.DataFrame([[1, 2], [3, 4], [5, 6], [7, 8], [9, 0]])
    size = 100000
    test = pd.DataFrame({0: list(range(size)), 1: list(range(size))})
    test = test % 21
    print(test)
    test[0] = apply(test[0], test_fun)
    print(test)
