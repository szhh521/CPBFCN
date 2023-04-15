import os
import h5py
import os.path as osp
import numpy as np
import random
import torch
from torch.utils import data

__all__ = ['CRIDataSetTrain', 'CRIDataSetTest']


class CRIDataSetTrain(data.Dataset):
    def __init__(self, data_tr, denselabel_tr):
        super(CRIDataSetTrain, self).__init__()
        self.data = data_tr
        self.denselabel = denselabel_tr

        assert len(self.data) == len(self.denselabel), \
            "the number of sequences and labels must be consistent."

        print("The number of positive data is {}".format(len(self.denselabel)))
        print("pre-process data is done.")

    def __len__(self):
        return len(self.denselabel)

    def __getitem__(self, index):
        data_one = self.data[index]
        denselabel_one = self.denselabel[index]

        return {"data": data_one, "denselabel": denselabel_one}


class CRIDataSetTest(data.Dataset):
    def __init__(self, data_te, denselabel_te):
        super(CRIDataSetTest, self).__init__()
        self.data = data_te
        self.denselabel = denselabel_te

        assert len(self.data) == len(self.denselabel), \
            "the number of sequences and labels must be consistent."
        print("The number of positive data is {}".format(len(self.denselabel)))
        print("pre-process data is done.")

    def __len__(self):
        return len(self.denselabel)

    def __getitem__(self, index):
        data_one = self.data[index]
        denselabel_one = self.denselabel[index]

        return {"data": data_one, "denselabel": denselabel_one}


