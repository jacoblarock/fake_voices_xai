from keras._tf_keras.keras.callbacks import History
import numpy as np
import matplotlib.pyplot as plt
import pickle

def get_history_stats(path: str) -> dict[str,np.ndarray]:
    out: dict[str,np.ndarray] = {}
    with open(path, "rb") as file:
        histories: list[History] = pickle.load(file)
    for history in histories:
        for key in history.history.keys():
            if key not in out.keys():
                out[key] = np.array([[0 for i in history.history[key]]])
            out[key] = np.concatenate((out[key], [history.history[key]]))
    return out

def plot_history(history_stats: dict[str,np.ndarray],
                 stat: str,
                 epoch: int
                 ) -> None:
    plt.plot(history_stats[stat][:,epoch])
    plt.show()

def find_min(history_stats: dict[str,np.ndarray],
             stat: str,
             epoch: int
             ) -> tuple:
    temp = history_stats[stat][:,epoch]
    min_stat = temp[1]
    print(len(temp))
    i = 1
    min_i = 1
    for stat in temp[2:]:
        if stat < min_stat:
            min_stat = stat
            min_i = i
        i += 1
    return (min_i, min_stat)

if __name__ == "__main__":
    test = get_history_stats("models/ItW_multi_percep_wval_cterm_u10000_histories")
    print(test)
    print(find_min(test, "val_loss", 0))
    plot_history(test, "val_loss", 0)
