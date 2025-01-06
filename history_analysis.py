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

if __name__ == "__main__":
    test = get_history_stats("trained_models/ItW_multi_percep_u10000e2/ItW_multi_percep_u10000e2_histories")
    print(test)
    plot_history(test, "loss", 0)
