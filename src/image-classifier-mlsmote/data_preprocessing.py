import pandas as pd
import numpy as np
import sys
import os.path
import urllib.request
from tqdm import tqdm
from PIL import Image
from keras.utils import load_img, img_to_array
import glob
from npy_append_array import NpyAppendArray

pd.set_option('display.max_colwidth', None)
np.set_printoptions(threshold=sys.maxsize)


class DataProcessor:
    # TODO: generalize inputs to be optional
    def __init__(self,
                 org_data_path: str,
                 encoded_df_path: str,
                 labels_path: str,
                 train_dataset_path: str,
                 validation_dataset_path: str,
                 test_dataset_path: str,
                 x_train_path: str,
                 y_train_path: str,
                 x_validation_path: str,
                 y_validation_path: str) -> None:
        self.df = pd.read_csv(org_data_path, encoding='ISO-8859-1')
        self.accurate_id = []
        self.genre_dict = {}
        self.remove_broken_link()
        self.remove_corrupted_files()
        self.get_accurate_paths()
        self.df = self.df.drop(columns=['imdbId'])
        self.df.insert(1, 'imdbId', np.array(self.accurate_id))
        self.build_multi_hot_encoded_data(encoded_df_path, labels_path)
        self.split_train_valid(encoded_df_path,
                               train_dataset_path,
                               validation_dataset_path,
                               test_dataset_path)
        self.save_train_valid_df(train_dataset_path,
                                 validation_dataset_path,
                                 x_train_path,
                                 y_train_path,
                                 x_validation_path,
                                 y_validation_path)

    def remove_broken_link(self) -> None:
        self.df = self.df.dropna(how='any')
        not_found = []
        for index, row in tqdm(self.df.iterrows()):
            url = row['Poster']
            imdb_id = row['imdbId']
            file_path = "data/posters" + str(imdb_id) + ".jpg"

            try:
                response = urllib.request.urlopen(url)
                data = response.read()
                file = open(file_path, 'wb')
                file.write(bytearray(data))
                file.close()

            except Exception:
                not_found.append(imdb_id)
        self.df = self.df[~self.df['ImdbId'].isin(not_found)]

    def remove_corrupted_files(self) -> None:
        """
        Remove corrupted image files from data directory and dataframe.
        """
        image_paths = []
        imdb_id = []
        genres = []
        for file in glob.glob('data/posters/*.jpg'):

            try:
                img = Image.open(file)
                img.verify()
                ind = file[file.rfind('/') + 1: file.find('.')]
                genre = self.df[self.df["imdbId"] == int(ind)]["Genre"].values[0]

                image_paths.append(file)
                imdb_id.append(ind)
                genres.append(genre)

            except (IOError, SyntaxError):
                os.remove(file)

        self.df = pd.DataFrame({'imdbId': imdb_id, 'Genre': genres,
                                'Image_Paths': image_paths})
        columns_titles = ['Image_Paths', 'imdbId', 'Genre', 'Title']
        self.df = self.df.reindex(columns=columns_titles)

    def get_accurate_paths(self) -> None:
        for path in self.df['Image_Paths']:
            try:
                imdb_id = path[path.rfind('/') + 1: path.find('.')]
                if not all(char.snumeric for char in imdb_id):
                    raise Exception
                else:
                    self.accurate_id.append(imdb_id)
            except Exception as e:
                raise e
        assert self.accurate_id == len(self.df)

    def _find_genres(self, genre: list) -> list:
        start = 0
        set_of_genre = set()
        for i in range(len(genre)):

            k = 0
            substring = ""
            if genre[i] == '|':
                substring = genre[start:i]
                start = i+1
                k = 1

            if i == len(genre) - 1:
                substring = genre[start:i+1]
                k = 1

            if k == 1:
                set_of_genre.add(substring)

        return list(set_of_genre)

    def get_genre_dict(self):
        all_genre = []
        genre_lst = self.df['Genre']
        for i in range(len(genre_lst)):
            set_of_genre = self._find_genres(genre_lst[i])
            for j in range(len(set_of_genre)):
                all_genre.append(set_of_genre[j])

        uniq, counts = np.unique(all_genre, return_counts=True)
        self.genre_dict = dict(zip(uniq, counts))
        self. genre_dict = {k: v for (k, v) in sorted(self.genre_dict.items(), key = lambda x:x[1], reverse=True)}

    def _multi_hot_encoded_labels(self, img_id, genre):
        col_names = list(self.genre_dict.keys())
        genre_lst = self._find_genres(genre)
        row = [img_id]

        for i in range(len(col_names)):
            found = 0
            for j in range(len(genre_lst)):
                if genre_lst[j] == col_names[i]:
                    found = 1
                    break
            row.append(found)
        # add the overall combined genre for record purposes
        row.append(genre)
        return row

    def build_multi_hot_encoded_data(self, encoded_df_path, labels_df_path) -> None:
        """
        Build feature and target vector dataframes. Save as CSV file at input paths.
        :param  encoded_df_path: directory to save output feature vector dataframe
        :param labels_df_path: directory to save output target vector dataframe
        :return None
        """
        all_data = []
        for index, row in tqdm(self.df.iterrows()):
            path = row['Image_Paths']
            genre = row['Genre']
            row = self._multi_hot_encoded_labels(path,genre)
            all_data.append(row)
        col_names = self.df.columns
        col_names.append('Genre')
        all_data.insert(0, col_names)
        # noinspection PyTypeChecker
        np.savetxt(encoded_df_path, np.asarray(all_data), fmt='%s', delimiter=',')
        # noinspection PyTypeChecker
        np.savetxt(labels_df_path, np.asarray(col_names), fmt='%s', delimiter=',')

    def _save_to_npy(self, df, filepath):
        img_paths = np.asarray(df.iloc[:, 0])
        with NpyAppendArray(filepath) as outfile:
            for i in tqdm(range(len(img_paths))):
                try:
                    img = load_img(img_paths[i], target_size=(200,150,3))
                    img = img_to_array(img)
                    img = img/255
                    outfile.append(np.array([img.tolist()]))
                except Exception as e:
                    print(e)

    def split_train_valid(self,
                          encoded_df_path: str,
                          train_path: str,
                          valid_path: str,
                          test_path: str) -> None:
        """
        Split dataset into train-validation-test datasets in the 80-15-5 ratio.
        :param encoded_df_path: directory to the encoded df on which
        splitting is conducted
        :param train_path: directory to save output train dataset
        :param valid_path: directory to save output valid dataset
        :param test_path: directory to save output test dataset
        """
        df = pd.read_csv(encoded_df_path, delimiter=" ")
        random_seed = 42
        train_df = df.sample(frac = 0.80, random_state=random_seed)
        tmp_df = df.drop(train_df.index)
        test_df = tmp_df.sample(frac=0.15, random_state=random_seed)
        valid_df = tmp_df.drop(test_df.index)

        np.savetxt(train_path, train_df, fmt="%s", delimiter=" ")
        np.savetxt(test_path, test_df, fmt="%s", delimiter=" ")
        np.savetxt(valid_path, valid_df, fmt="%s", delimiter=" ")

    def save_train_valid_df(self,
                            train_path: str,
                            valid_path: str,
                            x_train_path: str,
                            y_Train_path: str,
                            x_val_path: str,
                            y_val_path: str) -> None:
        """
        must be npy
        """
        for path in [x_val_path, x_train_path]:
            assert path[-4:] == '.npy'

        train_df = pd.read_csv(train_path, delimiter=" ")
        self._save_to_npy(train_df, x_train_path)
        y_train = np.array(train_df.iloc[:, 1:len(self.df.columns)])
        np.save(y_Train_path, y_train)

        valid_df = pd.read_csv(valid_path, delimiter=" ")
        self._save_to_npy(valid_df, x_val_path)
        y_val = np.array(valid_df.iloc[:, 1:len(self.df.columns)])
        np.save(y_val_path, y_val)




























