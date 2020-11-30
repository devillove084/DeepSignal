from torch.utils.data import dataset
import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset

class newTempData(Dataset):
    def __init__(self, filename, image_dir, width, height, repeat=False) -> None:
        self.filename = filename
        self.image_dir = image_dir
        self.width = width
        self.height = height
        self.repeat = repeat
        self.image_list = self.readfile(filename)
        self.len_image_list = len(self.readfile(filename))


    def __len__(self):
        if self.repeat == None:
            self.repeat = 100000
        else:
            data_len = 

    def __getitem__(self, index) :
        pass

    def readfile(path):
        image_label_list = list()
        with open(path, "r") as f :
            lines = f.readlines()
            for i in lines:
                

class generateData():
    def __init__(self) -> None:
        pass

    def generate():
        pass

    def generateLabel():
        pass
