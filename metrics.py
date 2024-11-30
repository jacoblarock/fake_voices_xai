from ast import literal_eval
from scipy.optimize import minimize

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
            result = 0 if summary["result"] < threshold else 1
            if res["label"] == 0:
                if res["result"] == 0:
                    summary["tn"] += 1
                if res["result"] == 1:
                    summary["fp"] += 1
            if res["label"] == 1:
                if res["result"] == 0:
                    summary["fn"] += 1
                if res["result"] == 1:
                    summary["tp"] += 1
        return (summary["tp"]+summary["tn"]) / (summary["fp"]+summary["fn"])
    # overload for summarized data
    elif type(data) == dict:
        return (data["tp"]+data["tn"]) / (data["fp"]+data["fn"])
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

if __name__ == "__main__":
    results = load_results("./trained_models/ItW_multi_percep_until10000_results.txt")
    eer = eer(results)
    print(eer)
