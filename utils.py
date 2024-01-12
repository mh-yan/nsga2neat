import numpy as np

def normalize(x, b=None, u=None):
    if b != None:
        x -= b
    else:
        x -= np.min(x)
    with np.errstate(divide='ignore', invalid='ignore'):
        if u != None:
            x /= u
        else:
            x /= np.max(x)
    x = np.nan_to_num(x)
    x *= 2
    x -= 1
    return x


def scale(x, min=0, max=1):
    # scale x to [min, max]
    x = np.nan_to_num(x)
    x = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-10)
    return x

from pathlib import Path

def create_folder(folder_path):
    folder = Path(folder_path)
    folder.mkdir(parents=True, exist_ok=True)
    # print(f"文件夹 '{folder_path}' 创建成功")

