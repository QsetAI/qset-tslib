import numpy as np


def argmin(arr, last=False, *args, **kwargs):
    # https://stackoverflow.com/questions/8768540/how-to-find-last-occurrence-of-maximum-value-in-a-numpy-ndarray
    if not last:
        return np.argmin(arr, *args, **kwargs)
    else:
        b = arr[::-1]
        return len(b) - np.argmin(b, *args, **kwargs) - 1


def test_argmin():
    arr = np.array([3, 1, 4, 0, 0, 6])
    print(argmin(arr))
    print(argmin(arr, last=True))


if __name__ == "__main__":
    test_argmin()
