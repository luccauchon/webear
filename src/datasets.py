from torch.utils.data import Dataset
from functools import lru_cache
import tsaug
import torch


class TripleIndicesDataset(Dataset):
    def __init__(self, _df, _indices, _feature_cols, _target_col, _device, _x_seq_length, _y_seq_length, _x_cols_to_norm=None, _y_cols_to_norm=None, mode='train', _normalize_by_first_row=True):
        """

        """
        self.add_noise      = False
        self.device         = _device
        self.df             = _df.copy()
        self.feature_cols   = _feature_cols.copy()
        self.indices        = _indices
        self.mode           = mode
        self.normalize_by_first_row = _normalize_by_first_row
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
        the_X, the_y = self.df.iloc[i1:i2], self.df.iloc[i2:i3]
        assert len(the_X) == self.x_seq_length and len(the_y) == self.y_seq_length

        x_data_norm, y_data_norm = None, None
        if self.normalize_by_first_row: # normalize all rows by the first row
            assert self.x_cols_to_norm is not None
            x_data_norm = the_X[self.x_cols_to_norm].iloc[0]
            y_data_norm = the_X[self.y_cols_to_norm].iloc[0]
            the_X = the_X[self.feature_cols].apply(lambda x: x.div(x_data_norm[x.name]) if x.name in x_data_norm else x)
            the_y = the_y[self.target_col].apply(lambda y:   y.div(y_data_norm[y.name]) if y.name in y_data_norm else y)

        if self.mode == 'train' and self.add_noise:
            the_X = (tsaug.AddNoise(scale=0.01, normalize=False) @ 0.5).augment(the_X.values)

        the_X = torch.tensor(the_X.values, dtype=torch.float, device=self.device)
        the_y = torch.tensor(the_y.values, dtype=torch.float, device=self.device)

        if self.normalize_by_first_row:
            x_data_norm = torch.tensor(x_data_norm.values, dtype=torch.float, device=self.device)
            y_data_norm = torch.tensor(y_data_norm.values, dtype=torch.float, device=self.device)

        return the_X, the_y, x_data_norm, y_data_norm