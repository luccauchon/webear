from torch.utils.data import Dataset
from functools import lru_cache
from loguru import logger
import tsaug
import torch
import numpy as np


class TripleIndicesRegressionDataset(Dataset):
    def __init__(self, _df, _indices, _feature_cols, _target_col, _device, _x_seq_length, _y_seq_length, _x_cols_to_norm=None, _y_cols_to_norm=None, mode='train', _normalize_by_first_row_and_center_at_0=True, _add_noise=False):
        """

        """
        self.add_noise      = _add_noise
        self.device         = _device
        self.df             = _df.copy()
        self.feature_cols   = _feature_cols.copy()
        self.indices        = _indices
        self.mode           = mode
        self.normalize_by_first_row_and_center_at_0 = _normalize_by_first_row_and_center_at_0
        self.target_col     = _target_col.copy()
        self.x_cols_to_norm = _x_cols_to_norm
        self.y_cols_to_norm = _y_cols_to_norm
        self.x_seq_length   = _x_seq_length
        self.y_seq_length   = _y_seq_length

    def __len__(self):
        return len(self.indices)

    @lru_cache(maxsize=None)
    def __getitem__(self, idx):
        i1, i2, i3   = self.indices[idx]
        the_x, the_y = self.df.iloc[i1:i2], self.df.iloc[i2:i3]
        assert len(the_x) == self.x_seq_length and len(the_y) == self.y_seq_length

        # if self.mode == 'train' and self.add_noise:
        #     the_x = the_x[self.feature_cols].apply(lambda x: x.np.random.normal(0,1) if x.name in x_data_norm else x)

        x_data_norm, y_data_norm = None, None
        if self.normalize_by_first_row_and_center_at_0: # normalize all rows by the first row
            assert self.x_cols_to_norm is not None
            x_data_norm = the_x[self.x_cols_to_norm].iloc[0]
            y_data_norm = the_x[self.y_cols_to_norm].iloc[0]
            the_x = the_x[self.feature_cols].apply(lambda x: x.div(x_data_norm[x.name]) if x.name in x_data_norm else x)
            the_x = the_x[self.feature_cols].apply(lambda x: x - 1 if x.name in x_data_norm else x)
            the_y = the_y[self.target_col].apply(lambda y:   y.div(y_data_norm[y.name]) if y.name in y_data_norm else y)
            the_y = the_y[self.target_col].apply(lambda y: y - 1 if y.name in y_data_norm else y)

        if self.mode == 'train' and self.add_noise:
            the_x = (tsaug.AddNoise(scale=0.01, normalize=False) @ 0.5).augment(the_x.values)

        try:
            the_x = torch.tensor(the_x.values, dtype=torch.float, device=self.device)
        except:
            the_x = torch.tensor(the_x, dtype=torch.float, device=self.device)
        the_y = torch.tensor(the_y.values, dtype=torch.float, device=self.device)

        if self.normalize_by_first_row_and_center_at_0:
            x_data_norm = torch.tensor(x_data_norm.values, dtype=torch.float, device=self.device)
            y_data_norm = torch.tensor(y_data_norm.values, dtype=torch.float, device=self.device)

        return the_x, the_y, x_data_norm, y_data_norm


class TripleIndicesLookAheadClassificationDataset(Dataset):
    def __init__(self, _df, _feature_cols, _target_col, _device, _x_cols_to_norm, _indices, _mode, _margin, _direction_of_ones="up", _data_augmentation=False, power_of_noise=0.001):
        """
        Args:
            df (pd.DataFrame): The input DataFrame.
            feature_cols (list): A list of column names for the features.
            target_col (str): The column name for the target variable.
        """
        self.df             = _df.copy()
        self.feature_cols   = _feature_cols.copy()
        self.target_col     = _target_col.copy()
        self.x_cols_to_norm = _x_cols_to_norm.copy()
        self.device         = _device
        self.indices        = _indices
        self.mode           = _mode
        self.data_augmentation = _data_augmentation
        self.margin         = _margin
        self.direction_of_ones  = _direction_of_ones
        self.power_of_noise     = power_of_noise
        self.frequency_of_noise = 0.25
        logger.info(f"[{self.mode}] Using a margin of {self.margin}")

    def __len__(self):
        return len(self.indices)

    @lru_cache(maxsize=None)
    def __getitem__(self, idx):
        i1, i2, i3 = self.indices[idx]
        the_x, the_y = self.df.iloc[i1:i2], self.df.iloc[i2:i3]

        assert 1 == len(the_y)
        vx, vy = the_x.iloc[-1][self.target_col].values[0], the_y.iloc[-1][self.target_col].values[0]

        if self.direction_of_ones == 'up':
            the_target = 1 if vy > vx + self.margin else 0  # "1" if direction is up
        else:
            the_target = 1 if vy < vx + self.margin else 0  # "1" if direction is down

        # normalize all rows by the first row
        x_data_norm = the_x[self.x_cols_to_norm].iloc[0]
        if self.mode == 'train' and (self.data_augmentation and np.random.normal(0, 1)>1.-self.frequency_of_noise):
            the_x = the_x[self.feature_cols].apply(lambda x: x.div(x_data_norm[x.name]) + np.random.normal(0, self.power_of_noise, size=len(x)) if x.name in x_data_norm else x)
        elif self.mode == 'inference':
            if self.data_augmentation:
                the_x = the_x[self.feature_cols].apply(lambda x: x.div(x_data_norm[x.name]) + np.random.normal(0, self.power_of_noise, size=len(x)) if x.name in x_data_norm else x)
            else:
                the_x = the_x[self.feature_cols].apply(lambda x: x.div(x_data_norm[x.name]) if x.name in x_data_norm else x)
        else:
            the_x = the_x[self.feature_cols].apply(lambda x: x.div(x_data_norm[x.name]) if x.name in x_data_norm else x)

        x = torch.tensor(the_x.values, dtype=torch.float, device=self.device)
        y = torch.tensor(the_target,   dtype=torch.float, device=self.device)

        x_data_norm = torch.tensor(x_data_norm.values, dtype=torch.float, device=self.device)

        return x, y, x_data_norm