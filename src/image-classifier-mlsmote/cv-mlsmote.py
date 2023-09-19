import pandas as pd
import numpy as np
import tqdm
import io
import sys
import os.path
import urllib.request
from tqdm import tqdm
from os import listdir
import glob
import matplotlib.pyplot as plt
import random
from sklearn.datasets import make_classification
from sklearn.neighbors import NearestNeighbors
from data_loader import DataLoader
import csv
from npy_append_array import NpyAppendArray
from distributed import Client
from dask.distributed import wait
from cuml.dask.common.utils import persist_across_workers
import dask_cudf
import cupy as cp
import cudf
from cudf import DataFrame
from cuml.dask.neighbors import NearestNeighbors
import os



pd.set_option('display.max_colwidth', None)
np.set_printoptions(threshold=sys.maxsize)


class MLSMOTE:
    def __init__(self, feature_df_path, target_df_path, num_sample: int):
        """
        feature_df_path: directory to feature vector DataFrame
        target_df_path: directory to target vector DataFrame
        n_sample: int, number of newly generated samples
        """
        self.feature_df = pd.read_csv(feature_df_path)
        self.target_df = pd.read_csv(target_df_path)
        self.num_sample = num_sample

    def get_minority_label(self) -> list:
        """
        Calculating Imbalance Ratio per label (IRPL) and Mean Imbalance ratio
        (MIR), defined as the average of IRPL of all the labels.
        Find minority labels of the given target dataframe.
        :return:
        """
        columns = self.target_df.columns
        n = len(columns)
        count = []

        for col in range(n):
            count.append(int(self.target_df[columns[col]].value_counts()[1]))

        count_dict = {'Label': [], 'Count': []}
        for i in range(n):
            count_dict['Label'].append(columns[i])
            count_dict['Count'].append(count[i])

        count_df = pd.DataFrame.from_dict(count_dict)

        irpl = [max(count)/c for c in count]
        irpl_dict = {'Label': [], 'IRPL': []}
        for i in range(n):
            irpl_dict['Label'].append(columns[i])
            irpl_dict['IRPL'].append(irpl[i])
        irpl_df = pd.DataFrame.from_dict(irpl_dict)

        mir = np.average(irpl)
        print("\n Mean Imbalance Ratio: ", mir)

        minority_label = []
        for i in range(n):
            if irpl[i] > mir:
                minority_label.append(columns[i])

        print("Minority Labels: ", minority_label)
        return count_dict, irpl_dict, minority_label

    def get_num_minority_instances(self, minority_label, count_dict):
        # TODO: return dict of minority labels
        for i in range(len(minority_label)):
            for genre in count_dict['Label']:
                if minority_label[i] == genre:
                    genre_ind = count_dict['Label'].index(genre)
                    print(minority_label[i], count_dict["Count"][genre_ind])

    def flatten_img(self, df, filename: str):
        with NpyAppendArray(filename) as outfile:
            for i in tqdm(range(len(df))):
                try:
                    img_flatten = df[i].flatten()
                    outfile.append(np.array([img_flatten.tolist()]))
                except Exception as e:
                    print('Error', e)

    def config_Dask_CUDA(self) -> None:
        """
        TODO: consider moving to new module for scaling
        """
        os.environ["DASK_RMM__POOL_SIZE"] = '500M'
        os.environ["DASK_UCX__CUDA_COPY"] = "True"
        os.environ["DASK_UCX__TCP"] = "True"
        os.environ["DASK_UCX__NVLINK"] = "True"
        os.environ["DASK_UCX__INFINIBAND"] = "True"
        os.environ["DASK_UCX__NET_DEVICES"] = "ib0"

    def nearest_neighbor(self, X: np.array) -> list:
        """
        Give index of 5 nearest neighbor of all the instances
        : param X: numpy array of features whose nearest neighbor has to find
        : return indices: list of list, index of 5 nearest neighbor of each
        element in X
        """


    def MLSMOTE(self) -> [pd.DataFrame, pd.DataFrame]:
        """
        Generate the augmented data using MLSMOTE algorithm.
        : return new_X: pd.DataFrame, augmented feature vector data
        : target: pd.FataFrame, augmented target vector data
        """
        indices2 = nearest_neighbour(X)
        n = len(indices2)
        new_X = np.zeros((self.num_sample, X.shape[1]))
        target = np.zeros((n_sample, y.shape[1]))
        for i in range(n_sample):
            reference = random.randint(0,n-1)
            neighbour = random.choice(indices2[reference,1:])
            all_point = indices2[reference]
            nn_df = y[y.index.isin(all_point)]
            ser = nn_df.sum(axis = 0, skipna = True)
            target[i] = np.array([1 if val>2 else 0 for val in ser])
            ratio = random.random()
            gap = X.loc[reference,:] - X.loc[neighbour,:]
            new_X[i] = np.array(X.loc[reference,:] + ratio * gap)
        new_X = pd.DataFrame(new_X, columns=X.columns)
        target = pd.DataFrame(target, columns=y.columns)
        new_X = pd.concat([X, new_X], axis=0)
        target = pd.concat([y, target], axis=0)
        return new_X, target

def main():
    # TODO: move to new module solely for driver code
    df = pd.read_csv("/content/drive/MyDrive/Summer_2023/Models/Multi_hot_encoded_data.csv", delimiter=",")
    count_dict, irpl_dict, minority_label = get_minority_label(df, 1, 28)
    get_num_minority_instances(minority_label, count_dict)

    X_Train = np.load('/content/drive/MyDrive/Summer_2023/Models/Poster_Models/X_Train.npy', mmap_mode = 'r')
    Y_Train = np.load('/content/drive/MyDrive/Summer_2023/Models/Poster_Models/Y_Train.npy', mmap_mode = 'r')

    mlsmote_train_filename = '/content/drive/MyDrive/Summer_2023/Models/Poster_Models/MLSMOTE/mlsmote_X_Train_flaten.npy'
    flatten_img(X_Train, mlsmote_train_filename)
    mlsmote_y_train_filename = '/content/drive/MyDrive/Summer_2023/Models/Poster_Models/MLSMOTE/mlsmote_Y_Train.npy'
    mlsmote_Y_Train_df = Y_Train[1:]
    mlsmote_X_Train = np.load(mlsmote_train_filename, mmap_mode = 'r')

    # Connect to a cluster through a Dask client
    client = Client(scheduler_file='dask-scheduler.json')
    X_cudf = cudf.DataFrame(mlsmote_X_Train)




