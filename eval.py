import classification
import os

if __name__ == "__main__":
    labels = classification.get_labels("./datasets/ASVspoof2021_DF_eval/DF/CM/trial_metadata.txt", 1, 5, "spoof", "bonafide", delimiter=" ")
    print(labels)
