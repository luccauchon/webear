from torch.utils.data import Dataset
import datetime
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


class TripleIndicesLookAheadBinaryClassificationDataset(Dataset):
    def __init__(self, _df, _feature_cols, _target_col, _device, _x_cols_to_norm, _indices, _mode, _margin, _type_of_margin, _direction,
                 _data_augmentation=False, _power_of_noise=0.001, _frequency_of_noise=0.25, _just_x_no_y=False):
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
        self.type_of_margin = _type_of_margin
        self.power_of_noise     = _power_of_noise
        self.frequency_of_noise = _frequency_of_noise
        self.just_x_no_y        = _just_x_no_y
        self._direction = _direction
        if self.mode in ['train', 'test', 'val']:
            if self.type_of_margin == 'relative':
                logger.debug(f"[{self.mode}] Using a relative margin of {self.margin}% , direction is {self._direction}")
            if self.type_of_margin == 'fixed':
                logger.debug(f"[{self.mode}] Using a fixed margin of {self.margin} , direction is {self._direction}")
        if self.mode in ['test', 'val']:
            assert not self.data_augmentation

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.just_x_no_y:
            i1, i2 = self.indices[idx]
            the_x = self.df.iloc[i1:i2]

            # normalize all rows by the first row
            x_data_norm = the_x[self.x_cols_to_norm].iloc[0]
            if self.mode == 'train':
                assert False
            elif self.mode == 'inference':
                if self.data_augmentation:
                    if 0 != len(x_data_norm):
                        the_x = the_x[self.feature_cols].apply(lambda x: x.div(x_data_norm[x.name]) + np.random.normal(0, self.power_of_noise, size=len(x)) if x.name in x_data_norm else x)
                        assert False, f"to be verified"
                    else:
                        _toto = the_x[[a_col for a_col in self.feature_cols if a_col != 'day_of_week']].iloc[0]
                        the_x = the_x[self.feature_cols].apply(lambda x: x + np.random.normal(0, self.power_of_noise, size=len(x)) if x.name in _toto else x)
                else:
                    the_x = the_x[self.feature_cols].apply(lambda x: x.div(x_data_norm[x.name]) if x.name in x_data_norm else x)
            else:
                assert False
            x = torch.tensor(the_x.values, dtype=torch.float, device=self.device)
            x_data_norm = torch.tensor(x_data_norm.values, dtype=torch.float, device=self.device)
            return x, -1., x_data_norm

        i1, i2, i3, i4 = self.indices[idx]
        the_x, the_y = self.df.iloc[i1:i2], self.df.iloc[i3:i4]

        prediction_date_milliseconds = int(datetime.datetime.combine(the_y.index[0].date(), datetime.time()).timestamp() * 1000)

        assert 1 == len(the_y)
        vx, vy = the_x.iloc[-1][self.target_col].values[0], the_y.iloc[-1][self.target_col].values[0]
        if self.type_of_margin == 'fixed':
            sunny_side = self.margin
        elif self.type_of_margin == 'relative':
            sunny_side = vx * (self.margin / 100.)
        else:
            assert False, f"{self.type_of_margin=}"
        if self.mode != 'inference':
            if 'up' == self._direction:
                the_target = 1 if vy > vx + sunny_side else 0
            else:
                the_target = 1 if vy < vx - sunny_side else 0
        else:
            if vy > vx + sunny_side:
                the_target = 1
            elif vy < vx - sunny_side:
                the_target = -1
            else:
                the_target = 0

        # normalize all rows by the first row
        x_data_norm = the_x[self.x_cols_to_norm].iloc[0]
        if self.mode == 'train' and (self.data_augmentation and np.random.uniform(0, 1)>1.-self.frequency_of_noise):
            if 0 != len(x_data_norm):
                the_x = the_x[self.feature_cols].apply(lambda x: x.div(x_data_norm[x.name]) + np.random.normal(0, self.power_of_noise, size=len(x)) if x.name in x_data_norm else x)
            else:
                _toto = the_x[[a_col for a_col in self.feature_cols if a_col != 'day_of_week']].iloc[0]
                the_x = the_x[self.feature_cols].apply(lambda x: x + np.random.normal(0, self.power_of_noise, size=len(x)) if x.name in _toto else x)
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
        return x, y, (x_data_norm, prediction_date_milliseconds)


class TripleIndicesLookAheadTernaryClassificationDataset(Dataset):
    def __init__(self, _df, _feature_cols, _target_col, _device, _x_cols_to_norm, _indices, _mode, _pct_sideways, _num_classes,
                 _data_augmentation=False, _power_of_noise=0.001, _just_x_no_y=False, _frequency_of_noise=0.25):
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
        self.pct_sideways  = _pct_sideways
        assert 0. < self.pct_sideways < 1.
        assert 3 == _num_classes
        self.power_of_noise     = _power_of_noise
        self.frequency_of_noise = _frequency_of_noise
        self.just_x_no_y        = _just_x_no_y
        if self.mode in ['train', 'test']:
            logger.info(f"[{self.mode}] Using sideways of {self.pct_sideways}%")
        if self.mode in ['test']:
            assert not self.data_augmentation

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.just_x_no_y:
            i1, i2 = self.indices[idx]
            the_x = self.df.iloc[i1:i2]

            # normalize all rows by the first row
            x_data_norm = the_x[self.x_cols_to_norm].iloc[0]
            if self.mode == 'train':
                assert False
            elif self.mode == 'inference':
                if self.data_augmentation:
                    the_x = the_x[self.feature_cols].apply(lambda x: x.div(x_data_norm[x.name]) + np.random.normal(0, self.power_of_noise, size=len(x)) if x.name in x_data_norm else x)
                else:
                    the_x = the_x[self.feature_cols].apply(lambda x: x.div(x_data_norm[x.name]) if x.name in x_data_norm else x)
            else:
                assert False
            x = torch.tensor(the_x.values, dtype=torch.float, device=self.device)
            x_data_norm = torch.tensor(x_data_norm.values, dtype=torch.float, device=self.device)
            return x, -1., x_data_norm

        i1, i2, i3 = self.indices[idx]
        the_x, the_y = self.df.iloc[i1:i2], self.df.iloc[i2:i3]

        assert 1 == len(the_y)
        vx, vy = the_x.iloc[-1][self.target_col].values[0], the_y.iloc[-1][self.target_col].values[0]
        sunny_side = vx * (self.pct_sideways/100.)
        if vy > vx + sunny_side:
            the_target = 2
        elif vy < vx - sunny_side:
            the_target = 0
        else:
            the_target = 1

        # normalize all rows by the first row
        x_data_norm = the_x[self.x_cols_to_norm].iloc[0]
        if self.mode == 'train' and (self.data_augmentation and np.random.uniform(0, 1)>1.-self.frequency_of_noise):
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