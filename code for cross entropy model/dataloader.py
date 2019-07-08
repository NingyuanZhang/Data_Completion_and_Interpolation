import pandas as pd
import os

class DataLoader(object):
    def __init__(self, path, header_path, sep=',', random_seed=2019):
        self.path = path    # raw data file path
        self.header_path = header_path  # training data X column names
        self.data_loaded = False    # mark if the data is loaded
        self.sep = sep      # the separator used in csv file: ',' '\t', etc.
        self.random_seed = random_seed
        self._load_data()
        self._x_col_name, self._y_col_name = self._load_header()

    def _load_data(self):
        """
        Load data from raw csv file. This is called by the init function
        :return: None
        """
        if os.path.exists(self.path) and self.data_loaded is False:
            self.data_df = pd.read_csv(self.path, sep=self.sep)
            self.data_loaded = True
        if os.path.exists(self.path) is False:
            raise Exception('Invalid path')

    def _load_header(self):
        if os.path.exists(self.header_path):
            data_df = pd.read_csv(self.header_path, sep=self.sep)
            x_cols = data_df.columns.values
            all_cols = self.data_df.columns.values
            y_cols = [x for x in all_cols if x not in x_cols]
            return x_cols, y_cols

    def get_xy_headers(self):
        return self._x_col_name, self._y_col_name

    def get_y_class_number(self):
        """
        Get all the number of classes information
        :return: a dictionary with y column name as the key and number as value
        """
        # num_of_classes = (self.data_df.max(axis=0) - self.data_df.min(axis=0) + 1).astype(int)
        num_of_classes = self.data_df.max(axis=0).astype(int)
        return num_of_classes.to_dict()

    def data_split(self, train_ratio, col_list):
        """
        Split data into train validation and test sets
        :param train_ratio: the ratio of training set among the entire data
        :param col_list: a list of column names to be used
        :return: train validation and test data frames
        """
        if self.data_loaded is False:
            self._load_data()
        df = self.data_df[col_list].copy()
        df = df.dropna()
        train_df = df.sample(frac=train_ratio, random_state=self.random_seed)
        if train_ratio < 1.0:
            remain = df.drop(train_df.index)
            valid_df = remain.sample(frac=0.5, random_state=self.random_seed)
            test_df = remain.drop(valid_df.index)
            return train_df, valid_df, test_df
        else:
            return train_df, None, None

    def process_np(self, data_frame, label='label'):
        if self.data_loaded is False:
            self._load_data()
        tmp_df = data_frame.drop(columns=[label])
        X = tmp_df.values
        Y = data_frame[label].astype('int64').values - 1
        return X, Y