from loguru import logger
import torch.nn as nn
import torch
from torch.autograd import Variable
import inspect
import torch.nn.functional as F


class LSTMMetaUpDownClassification(nn.Module):
    def __init__(self, up_models, down_models):
        super(LSTMMetaUpDownClassification, self).__init__()
        self.up_models = up_models
        self.down_models = down_models
        self.list_of_model_names = None

    def get_number_models(self):
        return len(self.models)

    def forward(self, x, y=None):
        up_predictions, down_predictions = [], []
        for name_model, a_model in self.up_models.items():
            prediction = a_model(x)
            up_predictions.append(torch.nn.Sigmoid()(prediction[0]))
        for name_model, a_model in self.down_models.items():
            prediction = a_model(x)
            down_predictions.append(torch.nn.Sigmoid()(prediction[0]))

        up_ones, up_total = torch.count_nonzero(torch.stack(up_predictions) >= 0.5).item(), torch.stack(up_predictions, dim=0).shape[0]
        dw_ones, dw_total = torch.count_nonzero(torch.stack(down_predictions) >= 0.5).item(), torch.stack(down_predictions, dim=0).shape[0]
        up, down = up_ones/up_total, dw_ones/dw_total

        if 1==up and 0==down:
            return 1
        if 0==up and 1==down:
            return -1
        if 0==up and 0==down:
            return 0
        return 99

    def get_corresponding_model_names(self):
        return self.list_of_model_names


class LSTMClassification(nn.Module):
    def __init__(self, seq_length, num_input_features, hidden_size, num_layers, bidirectional, device, dropout, pos_weight=None, binary_classification=True, nb_classes=-1):
        super(LSTMClassification, self).__init__()
        self.num_layers = num_layers  # number of layers
        assert 1 == self.num_layers
        self.num_input_features = num_input_features  # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length
        self.d = 2 if bidirectional else 1
        assert self.d == 1
        self.device = device
        self.lstm1 = nn.LSTM(input_size=num_input_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, proj_size=0, bidirectional=bidirectional, device=device)
        self.lin1_1 = nn.Linear(in_features=hidden_size, out_features=hidden_size//2, device=device)
        self.lin1_2 = nn.Linear(in_features=hidden_size, out_features=hidden_size // 2, device=device)
        self.lin1_3 = nn.Linear(in_features=hidden_size, out_features=hidden_size // 2, device=device)
        self.lstm2 = nn.LSTM(input_size=hidden_size//2, hidden_size=hidden_size//2, num_layers=num_layers, batch_first=True, proj_size=0, bidirectional=bidirectional, device=device)
        self.lin2_1 = nn.Linear(in_features=hidden_size//2, out_features=hidden_size//4, device=device)
        self.lin2_2 = nn.Linear(in_features=hidden_size // 2, out_features=hidden_size // 4, device=device)
        self.lin2_3 = nn.Linear(in_features=hidden_size // 2, out_features=hidden_size // 4, device=device)
        self.lstm3 = nn.LSTM(input_size=hidden_size//4, hidden_size=hidden_size//4, num_layers=num_layers, batch_first=True, proj_size=0, bidirectional=bidirectional, device=device)
        self.lin3_1 = nn.Linear(in_features=hidden_size // 4, out_features=hidden_size // 8, device=device)
        self.lin3_2 = nn.Linear(in_features=hidden_size // 4, out_features=hidden_size // 8, device=device)
        self.lin3_3 = nn.Linear(in_features=hidden_size // 4, out_features=hidden_size // 8, device=device)
        self.lstm4 = nn.LSTM(input_size=hidden_size // 8, hidden_size=hidden_size // 8, num_layers=num_layers, batch_first=True, proj_size=0, bidirectional=bidirectional, device=device)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.binary_classification = binary_classification
        self.nb_classes = nb_classes
        self.activation = torch.nn.Hardtanh(min_val=-4, max_val=4)
        self.mode = 0
        if self.binary_classification:
            self.loss_function = nn.BCEWithLogitsLoss(reduction='sum').to(device)
            if pos_weight is not None:
                logger.debug(f"Setting pos_weight to {pos_weight}")
                self.loss_function = nn.BCEWithLogitsLoss(reduction='sum',pos_weight=torch.tensor(pos_weight, device=device)).to(device)
            else:
                self.loss_function = nn.BCEWithLogitsLoss(reduction='sum', pos_weight=torch.tensor(1., device=device)).to(device)
            self.fc = nn.Linear(hidden_size // 8, 1, device=device)  # fully connected last layer
        else:
            self.loss_function = nn.CrossEntropyLoss(reduction='sum').to(device)
            self.fc = nn.Linear(hidden_size // 8, self.nb_classes, device=device)  # fully connected last layer
            self.mode = 1
            self.conv1d = torch.nn.Conv1d(self.seq_length, self.nb_classes, kernel_size=3, padding=1, stride=1, device=device)
            self.fc2 = nn.Linear(hidden_size // 8, 1, device=device)  # fully connected last layer
        self.relu = torch.nn.ReLU()

    def forward(self, x, y=None):
        batch_size = x.shape[0]
        h_0 = Variable(torch.zeros(self.d * self.num_layers, batch_size, self.hidden_size)).to(self.device)  # hidden state
        c_0 = Variable(torch.zeros(self.d * self.num_layers, batch_size, self.hidden_size)).to(self.device)  # internal state
        # Propagate input through LSTM
        logits, (hn, cn) = self.lstm1(x, (h_0, c_0))  # lstm with input, hidden, and internal state
        logits, (hn, cn) = self.lstm2(self.relu(self.dropout(self.lin1_1(logits))), (self.relu(self.dropout(self.lin1_2(hn))), self.relu(self.dropout(self.lin1_3(cn)))))
        logits, (hn, cn) = self.lstm3(self.relu(self.dropout(self.lin2_1(logits))), (self.relu(self.dropout(self.lin2_2(hn))), self.relu(self.dropout(self.lin2_3(cn)))))
        logits, (hn, cn) = self.lstm4(self.relu(self.dropout(self.lin3_1(logits))), (self.relu(self.dropout(self.lin3_2(hn))), self.relu(self.dropout(self.lin3_3(cn)))))
        assert hn.shape[0] == self.d * self.num_layers
        if self.binary_classification or 1 == self.mode:
            hn = hn[-1]
            output = self.fc(hn)
        else:
            logits = self.conv1d(logits)
            # logits = logits[:, 0:-1, :]  # Drop last, it is unstable
            logits = self.fc2(logits).squeeze()
            output = self.activation(logits)

        loss = None
        if y is not None:
            if self.binary_classification:
                if 1 == len(y.shape):
                    y = y.unsqueeze(1)
                assert output.shape == y.shape, f"{output.shape=}  {y.shape=}"
                loss = self.loss_function(output, y)
            else:
                y = F.one_hot(y.type(torch.int64), num_classes=self.nb_classes)
                assert output.shape == y.shape, f"{output.shape=}  {y.shape=}"
                loss = self.loss_function(output, y.float())

        return output, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay, 'lr': learning_rate},
            {'params': nodecay_params, 'weight_decay': 0.0, 'lr': learning_rate}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        logger.debug(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        logger.debug(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, betas=betas, **extra_args)

        return optimizer


class LSTMRegression(nn.Module):
    def __init__(self, num_output_vars, t_length_output_vars, seq_length, num_input_features, hidden_size, num_layers, bidirectional, device, activation_minmax, dropout):
        super(LSTMRegression, self).__init__()
        self.num_output_vars      = num_output_vars  # number of classes
        self.t_length_output_vars = t_length_output_vars  # sequence length @output
        self.num_layers = num_layers  # number of layers
        assert 1 == self.num_layers
        self.num_input_features = num_input_features  # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length
        self.d = 2 if bidirectional else 1
        assert 1 == self.d
        self.device = device

        self.lstm1 = nn.LSTM(input_size=num_input_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, proj_size=0, bidirectional=bidirectional, device=device)

        self.lin1_1 = nn.Linear(in_features=hidden_size * self.d, out_features=hidden_size//2, device=device)
        self.lin1_2 = nn.Linear(in_features=hidden_size * self.d, out_features=hidden_size // 2, device=device)
        self.lin1_3 = nn.Linear(in_features=hidden_size * self.d, out_features=hidden_size // 2, device=device)
        self.lstm2 = nn.LSTM(input_size=hidden_size//2, hidden_size=hidden_size//2, num_layers=num_layers, batch_first=True, proj_size=0, bidirectional=bidirectional, device=device)

        self.lin2_1 = nn.Linear(in_features=hidden_size//2  * self.d, out_features=hidden_size//4, device=device)
        self.lin2_2 = nn.Linear(in_features=hidden_size // 2  * self.d, out_features=hidden_size // 4, device=device)
        self.lin2_3 = nn.Linear(in_features=hidden_size // 2  * self.d, out_features=hidden_size // 4, device=device)
        self.lstm3 = nn.LSTM(input_size=hidden_size//4, hidden_size=hidden_size//4, num_layers=num_layers, batch_first=True, proj_size=0, bidirectional=bidirectional, device=device)

        self.lin3_1 = nn.Linear(in_features=hidden_size // 4  * self.d, out_features=hidden_size // 8, device=device)
        self.lin3_2 = nn.Linear(in_features=hidden_size // 4  * self.d, out_features=hidden_size // 8, device=device)
        self.lin3_3 = nn.Linear(in_features=hidden_size // 4  * self.d, out_features=hidden_size // 8, device=device)
        self.lstm4 = nn.LSTM(input_size=hidden_size // 8, hidden_size=hidden_size // 8, num_layers=num_layers, batch_first=True, proj_size=0, bidirectional=bidirectional, device=device)

        self.lin4_1 = nn.Linear(in_features=hidden_size // 8  * self.d, out_features=hidden_size // 16, device=device)
        self.lin4_2 = nn.Linear(in_features=hidden_size // 8  * self.d, out_features=hidden_size // 16, device=device)
        self.lin4_3 = nn.Linear(in_features=hidden_size // 8  * self.d, out_features=hidden_size // 16, device=device)
        self.lstm5 = nn.LSTM(input_size=hidden_size // 16, hidden_size=hidden_size // 16, num_layers=num_layers, batch_first=True, proj_size=0, bidirectional=bidirectional, device=device)

        self.lin5_1 = nn.Linear(in_features=hidden_size // 16  * self.d, out_features=hidden_size // 32, device=device)
        self.lin5_2 = nn.Linear(in_features=hidden_size // 16  * self.d, out_features=hidden_size // 32, device=device)
        self.lin5_3 = nn.Linear(in_features=hidden_size // 16  * self.d, out_features=hidden_size // 32, device=device)
        self.lstm6 = nn.LSTM(input_size=hidden_size // 32, hidden_size=hidden_size // 32, num_layers=num_layers, batch_first=True, proj_size=0, bidirectional=bidirectional, device=device)

        self.dropout = torch.nn.Dropout(p=dropout)
        # self.conv1d = torch.nn.Conv1d(self.seq_length, self.t_length_output_vars + 1,kernel_size=3, padding=1, stride=1, device=device)  # add a output that we will drop
        self.conv1d = torch.nn.Conv1d(self.seq_length, self.t_length_output_vars, kernel_size=3, padding=1, stride=1, device=device)  # add a output that we will drop
        self.fc = nn.Linear(hidden_size//32, self.num_output_vars, device=device)  # fully connected last layer
        self.activation = torch.nn.Hardtanh(min_val=activation_minmax[0], max_val=activation_minmax[1])
        # self.loss_function = torch.nn.SmoothL1Loss()
        self.loss_function = torch.nn.MSELoss(reduction='sum')  # mean-squared error for regression
        self.relu = torch.nn.LeakyReLU()

    def forward(self, x, y=None):
        batch_size = x.shape[0]
        h_0 = Variable(torch.zeros(self.d * self.num_layers, batch_size, self.hidden_size)).to(self.device)  # hidden state
        c_0 = Variable(torch.zeros(self.d * self.num_layers, batch_size, self.hidden_size)).to(self.device)  # internal state
        # Propagate input through LSTM
        logits, (hn, cn) = self.lstm1(x, (h_0, c_0))  # lstm with input, hidden, and internal state
        logits, (hn, cn) = self.lstm2(self.relu(self.dropout(self.lin1_1(logits))), (self.relu(self.dropout(self.lin1_2(hn))), self.relu(self.dropout(self.lin1_3(cn)))))
        logits, (hn, cn) = self.lstm3(self.relu(self.dropout(self.lin2_1(logits))), (self.relu(self.dropout(self.lin2_2(hn))), self.relu(self.dropout(self.lin2_3(cn)))))
        logits, (hn, cn) = self.lstm4(self.relu(self.dropout(self.lin3_1(logits))), (self.relu(self.dropout(self.lin3_2(hn))), self.relu(self.dropout(self.lin3_3(cn)))))
        logits, (hn, cn) = self.lstm5(self.relu(self.dropout(self.lin4_1(logits))), (self.relu(self.dropout(self.lin4_2(hn))), self.relu(self.dropout(self.lin4_3(cn)))))
        logits, (hn, cn) = self.lstm6(self.relu(self.dropout(self.lin5_1(logits))), (self.relu(self.dropout(self.lin5_2(hn))), self.relu(self.dropout(self.lin5_3(cn)))))
        assert hn.shape[0] == self.d * self.num_layers

        logits = self.conv1d(logits)
        # logits = logits[:, 0:-1, :]  # Drop last, it is unstable
        logits = self.fc(logits)
        output = self.activation(logits)

        loss = None
        if y is not None:
            assert output.shape == y.shape
            loss = self.loss_function(output, y)

        return output, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay, 'lr': learning_rate},
            {'params': nodecay_params, 'weight_decay': 0.0, 'lr': learning_rate}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        logger.info(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        logger.info(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, betas=betas, **extra_args)

        return optimizer