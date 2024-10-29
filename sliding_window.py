import numpy as np

"""
Class window: allows for easy sliding-window analysis of np arrays
"""
class window:

    """
    Constructor defines:
     - arr: the array to iterate over
     - window_length: the length of the window ahead of the current index
    """
    def __init__(self, arr: np.ndarray, window_length: int) -> None:
        self.i = 0
        self.window_length = window_length
        self.arr = arr
        return

    """
    hop the window by a factor given
    factor: percentage of window_length to hop
    returns 1 when the hop is made
    returns 0 when the hop could not be made, in which case the window moves to the final position
    """
    def hop(self, factor: float) -> int:
        if self.i + (self.window_length * factor) + self.window_length < len(self.arr):
            self.i += int(self.window_length * factor)
            return 1
        self.i = len(self.arr) - self.window_length - 1
        return 0

    """
    reset the position of the window to the beginning
    """
    def reset(self):
        self.i = 0

    """
    fetches the current window
    """
    def get_window(self) -> np.ndarray:
        window_indices = list(range(self.i, (self.i + self.window_length)))
        return self.arr[window_indices]
