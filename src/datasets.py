from torch.utils.data import Dataset
import datetime
from loguru import logger
import tsaug
import torch
import numpy as np
from utils import generate_indices_basic_style


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


class GenXandYRegressionData:
    """
    A class used to generate X and Y regression data.
    """

    def __init__(self, **kwargs):
        """
        Initializes the GenXandYRegressionData class.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        df = kwargs['df']
        the_dates = kwargs['the_dates']
        x_seq_length = kwargs['x_seq_length']
        y_seq_length = kwargs['y_seq_length']
        jump_ahead   = kwargs['jump_ahead']
        logger.debug(f"{x_seq_length=}   {y_seq_length=}   {jump_ahead=}")
        indices, df = generate_indices_basic_style(df=df.copy(), dates=the_dates, x_seq_length=x_seq_length, y_seq_length=y_seq_length, jump_ahead=jump_ahead)

        # Convert MultiIndex columns to simple columns by concatenating levels
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
        # Add this
        df['day_of_week'] = df.index.dayofweek + 1

        # Specify which columns are Xs and Ys
        x_cols_mutable = kwargs['x_cols_mutable']
        x_cols_fixed = kwargs['x_cols_fixed']
        y_cols = kwargs['y_cols']
        assert set(x_cols_mutable).issubset(df.columns)
        assert set(x_cols_fixed).issubset(df.columns)
        assert set(y_cols).issubset(df.columns)

        ###############################################################################
        # Ys
        ###############################################################################
        log_y_cols = [col + '_log' for col in y_cols]
        df[log_y_cols] = np.log(df[y_cols])

        # diff1_log_y_cols = [col + '_diff1' for col in log_y_cols]
        # df[diff1_log_y_cols] = df[log_y_cols].diff(1)
        # resulting_y_cols = diff1_log_y_cols
        resulting_y_cols = log_y_cols
        # # Reversing the operations
        # initial_log_values = df[log_y_cols].iloc[0]
        # reversed_log_values = df[diff1_log_y_cols].cumsum() + initial_log_values.values
        # reversed_original_values = np.exp(reversed_log_values)
        # assert np.allclose(reversed_original_values.iloc[-3:].values, df[y_cols].iloc[-3:].values)

        # Normalize
        norm_resulting_y_cols = [col + '_norm' for col in resulting_y_cols]
        self.ymean, self.ystd = df[resulting_y_cols].mean(), df[resulting_y_cols].std()
        df[norm_resulting_y_cols] = (df[resulting_y_cols] - self.ymean) / self.ystd
        self.norm_resulting_y_cols = norm_resulting_y_cols
        self.log_y_cols = log_y_cols
        #self.diff1_log_y_cols = diff1_log_y_cols
        np.allclose(df[y_cols].iloc[-1].values, self.denormalize_and_to_numpy(df[norm_resulting_y_cols].iloc[-1].values).values)
        ###############################################################################
        # Xs
        ###############################################################################
        # Log
        log_x_cols = [col + '_log' for col in x_cols_mutable]
        df[log_x_cols] = np.log(df[x_cols_mutable])

        # Rolling
        rolling3_x_cols = [col + f'_rolling3' for col in log_x_cols]
        df[rolling3_x_cols] = df[log_x_cols].rolling(window=3, center=True).mean()

        # Diff
        diff1_x_cols = [col + f'_diff1' for col in log_x_cols]
        df[diff1_x_cols] = df[log_x_cols].diff(1)

        # Concatenate all the Xs
        resulting_x_cols = log_x_cols + rolling3_x_cols + diff1_x_cols

        # Normalize
        norm_resulting_x_cols = [col + '_norm' for col in resulting_x_cols]
        self.xmean, self.xstd = df[resulting_x_cols].mean(), df[resulting_x_cols].std()
        df[norm_resulting_x_cols] = (df[resulting_x_cols] - self.xmean) / self.xstd
        self.norm_resulting_x_cols = norm_resulting_x_cols + x_cols_fixed

        self.mode = None
        split_nm_idx = 60
        self.train_indices, self.val_indices = indices[split_nm_idx:], indices[:split_nm_idx]
        self.df = df.copy()
        logger.debug(f"Take the last {split_nm_idx} time steps to do validation ({self.df.index[self.val_indices[-1][-1]].date()} to {self.df.index[self.val_indices[0][-1]].date()})")
        logger.debug(f"Training will use data from {self.df.index[self.train_indices[-1][-1]].date()} to {self.df.index[self.train_indices[0][-1]].date()}")

    def get_Xs(self):
        return self.norm_resulting_x_cols

    def get_Ys(self):
        return self.norm_resulting_y_cols

    def get_nb_in(self):
        return len(self.norm_resulting_x_cols)

    def get_nb_out(self):
        return len(self.norm_resulting_y_cols)

    def set_train(self):
        self.mode = 'train'
        return self

    def set_val(self):
        self.mode = 'val'
        return self

    def denormalize_and_to_numpy(self, value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
            result = np.exp((value * self.ystd.values) + self.ymean.values)
        else:
            result = np.exp((value * self.ystd) + self.ymean)
        return result

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_indices)
        else:
            return len(self.val_indices)

    def iterate(self):
        """
        Returns an iterator over the training ranges of df[norm_resulting_x_cols].

        Yields:
            pd.DataFrame: A slice of df[norm_resulting_x_cols] for each training range.
        """
        for idx in range(0, len(self)):
            yield self.get_item(idx)

    def get_item(self, idx):
        idx = self.train_indices[idx] if self.mode == 'train' else self.val_indices[idx]
        return self.df[self.norm_resulting_x_cols].iloc[idx[0]:idx[1]], self.df[self.norm_resulting_y_cols].iloc[idx[2]:idx[3]], self.df[self.log_y_cols].iloc[idx[2]:idx[3]]


class GenXandYRegressionDataset(Dataset):
    def __init__(self, grd):
        """

        """
        self.grd = grd

    def __len__(self):
        return len(self.grd)

    def __getitem__(self, idx):
        x, y, y_log = self.grd.get_item(idx)
        return x.values.astype(np.float32), y.values.astype(np.float32), y_log.values.astype(np.float32)