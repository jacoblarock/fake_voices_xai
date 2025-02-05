from ast import literal_eval
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np

def load_results(path: str) -> list | dict:
    data = open(path, "r").read()
    return literal_eval(data)

def accuracy(data: list | dict, threshold: float) -> float:
    """
    Calculate the accuracy of an evaluated model based on either an unsummarized list of results
    and a float threshold or a summarized list of results
    """
    if not (0 < threshold < 1):
        raise ValueError("threshold must be between 0 and 1")
    # overload for unsummarized data
    if type(data) == list:
        summary = {"tp": 0,
                   "tn": 0,
                   "fp": 0,
                   "fn": 0}
        for res in data:
            result = 0 if res["result"] < threshold else 1
            if res["label"] == 0:
                if result == 0:
                    summary["tn"] += 1
                if result == 1:
                    summary["fp"] += 1
            if res["label"] == 1:
                if result == 0:
                    summary["fn"] += 1
                if result == 1:
                    summary["tp"] += 1
        return (summary["tp"]+summary["tn"]) / (summary["tp"]+summary["tn"]+summary["fp"]+summary["fn"])
    # overload for summarized data
    elif type(data) == dict:
        return (data["tp"]+data["tn"]) / (data["tp"]+data["tn"]+data["fp"]+data["fn"])
    raise TypeError("data must be of type list or dict, a result of the evaluate method")

# function to minimize for calculation of EER
def _error_rate(x: float, data: list) -> float:
    total = len(data)
    misses = 0
    for res in data:
        result = 0 if res["result"] < x else 1
        if result != res["label"]:
            misses += 1
    return misses / total

def eer(data: list) -> dict:
    """
    Calculate the EER based on unsummarized results adjusting the threshold to minimize the error
    rate and returning the minimized error rate and corresponding threshold
    """
    if type(data) != list:
        raise TypeError("data must be of type list, an unsummarized result of the evaluate method")
    res = minimize(_error_rate, 0.5, (data), method="Nelder-Mead")
    threshold = res.x
    return {"eer": _error_rate(threshold, data), "threshold": threshold}

def tpr_fpr(data: list, threshold: float) -> tuple[float, float]:
    summary = {"tp": 0,
               "tn": 0,
               "fp": 0,
               "fn": 0}
    for res in data:
        result = 0 if res["result"] < threshold else 1
        if res["label"] == 0:
            if result == 0:
                summary["tn"] += 1
            if result == 1:
                summary["fp"] += 1
        if res["label"] == 1:
            if result == 0:
                summary["fn"] += 1
            if result == 1:
                summary["tp"] += 1
    tpr = summary["tp"] / (summary["tp"] + summary["fn"])
    fpr = summary["fp"] / (summary["fp"] + summary["tn"])
    return tpr, fpr

def gen_roc(data: list, save_path: str) -> None:
    x = []
    y = []
    for i in range(-10, 110):
        threshold = i / 100
        tpr, fpr = tpr_fpr(data, threshold)
        y.append(tpr)
        x.append(fpr)
    plt.clf()
    plt.rcParams.update({"font.size": 15})
    plt.ylim(0, max(y))
    plt.xlim(0, max(x))
    plt.plot(x, y, linewidth=3)
    plt.fill_between(x, y, 0, color="b", alpha=0.2)
    plt.savefig(save_path)

if __name__ == "__main__":
    import sys
    from glob import glob
    if len(sys.argv) > 1:
        threshold = float(sys.argv[1])
    else:
        threshold = 0.5
    result_list = glob("./trained_models/*/*results.txt")
    for result_path in result_list:
        print()
        model_name = result_path.split("/")[-2]
        print(model_name)
        result_dir = result_path[:result_path.rfind("/")+1]
        results = load_results(result_path)
        eer_res = eer(results)
        accuracy_res = accuracy(results, threshold)
        print("eer:", eer_res)
        print("accuracy:", accuracy_res)
        gen_roc(results, result_dir + "roc.png")
