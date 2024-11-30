from ast import literal_eval

def load_results(path: str) -> list | dict:
    data = open(path, "r").read()
    return literal_eval(data)

def accuracy(data: list | dict, threshold: float):
    if not (0 < threshold < 1):
        raise ValueError("threshold must be between 0 and 1")
    # overload for unsummarized data
    if type(data) == list:
        summary = {"tp": 0,
                   "tn": 0,
                   "fp": 0,
                   "fn": 0}
        for res in data:
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
