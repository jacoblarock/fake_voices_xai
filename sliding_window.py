import numpy as np

class window:
    """
    Class window: allows for easy sliding-window analysis of np arrays
    """

    def __init__(self, arr: np.ndarray, window_length: int, window_height: int = 10) -> None:
        """
        Constructor defines:
        - arr: the array to iterate over
        - window_length: the length of the window ahead of the current index
        """
        self.x = 0
        self.y = 0
        self.window_length = window_length
        self.window_height = window_height
        self.arr = arr
        if len(self.arr.shape) != 1:
            if self.arr.shape[0] < window_length:
                diff = window_length - self.arr.shape[0]
                self.arr = np.pad(self.arr, ((0, diff), (0, 0)))
            if self.arr.shape[1] < window_height:
                diff = window_height - self.arr.shape[1]
                self.arr = np.pad(self.arr, ((0, 0), (0, diff)))
        else:
            if self.arr.shape[0] < window_length:
                diff = window_length - self.arr.shape[0]
                self.arr = np.pad(self.arr, (0, diff))
        return

    def x_hop(self, factor: float) -> int:
        """
        hop the window by a factor given
        Argument:
        - factor: percentage of window_length to hop
        returns 1 when the hop is made
        returns 0 when the hop could not be made, in which case the window moves to the final position
        """
        if self.x + (self.window_length * factor) + self.window_length < len(self.arr):
            self.x += int(self.window_length * factor + 1)
            return 1
        self.x = len(self.arr) - self.window_length
        return 0

    def y_hop(self, factor: float) -> int:
        """
        hop the window vertically by a factor given
        Argument:
        - factor: percentage of window_height to hop
        returns 1 when the hop is made
        returns 0 when the hop could not be made, in which case the window moves to the final position
        """
        if self.y + (self.window_height * factor) + self.window_height < len(self.arr[0]):
            self.y += int(self.window_height * factor + 1)
            return 1
        self.y = len(self.arr[0]) - self.window_height
        return 0

    def smart_hop(self, factor: float) -> int:
        """
        hop the window vertically and horizontally by a factor given
        factor: percentage of window_length and window_height to hop
        returns 1 when the hop is made
        returns 0 when the hop could not be made, in which case the window moves to the final position
        """
        if len(self.arr.shape) == 1:
            return self.x_hop(factor)
        x_res = self.x_hop(factor)
        y_res = 1
        if x_res == 1:
            return 1
        y_res = self.y_hop(factor)
        if y_res == 0:
            return 0
        self.x = 0
        return 1

    def reset(self):
        """
        reset the position of the window to the beginning
        """
        self.x = 0
        self.y = 0

    def get_window(self) -> np.ndarray:
        """
        fetches the current window
        """
        if len(self.arr.shape) == 1 or self.window_height == 1:
            return self.arr[self.x:(self.x + self.window_length)]
        else:
            return self.arr[self.x:(self.x + self.window_length),
                            self.y:(self.y + self.window_height)]

if __name__ == "__main__":
    test_win = np.ndarray((20, 5))
    for x in range(test_win.shape[0]):
        for y in range(test_win.shape[1]):
            test_win[x, y] = x * 100 + y
    win = window(test_win, 10)
    res = 1
    print(win.x, win.y)
    print(win.get_window())
    print()
    while res == 1:
        res = win.smart_hop(0.5)
        print(win.x, win.y, win.arr.shape, len(win.arr[0]))
        print(win.get_window())
        print()
