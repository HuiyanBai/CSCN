import torch
from pathlib import *
import numpy as np
import random
import datetime
import os


def genConfusionMatrix(numClass, imgPredict, imgLabel):
    mask = (imgLabel != -1)
    label = numClass * imgLabel[mask] + imgPredict[mask]
    count = torch.bincount(label, minlength=numClass ** 2)
    confusionMatrix = count.reshape(numClass, numClass)
    return confusionMatrix


def set_seed(seed=43):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = (
            year + "_" + month + "_" + day + "_" + hour + "_" + minute + "_" + second
    )
    return time_filename