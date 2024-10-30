import numpy as np

class window:
    """
    Class window: allows for easy sliding-window analysis of np arrays
    """

    def __init__(self, arr: np.ndarray, window_length: int) -> None:
        """
        Constructor defines:
         - arr: the array to iterate over
         - window_length: the length of the window ahead of the current index
        """
        self.i = 0
        self.window_length = window_length
        self.arr = arr
        return

    def hop(self, factor: float) -> int:
        """
        hop the window by a factor given
        factor: percentage of window_length to hop
        returns 1 when the hop is made
        returns 0 when the hop could not be made, in which case the window moves to the final position
        """
        if self.i + (self.window_length * factor) + self.window_length < len(self.arr):
            self.i += int(self.window_length * factor)
            return 1
        self.i = len(self.arr) - self.window_length - 1
        return 0

    def reset(self):
        """
        reset the position of the window to the beginning
        """
        self.i = 0

    def get_window(self) -> np.ndarray:
        """
        fetches the current window
        """
        window_indices = list(range(self.i, (self.i + self.window_length)))
        return self.arr[window_indices]
