class PM1(nn.Module):

    def __init__(self, n_features_h, n_features_m, sequence_len_h, sequence_len_m, batch_size=64, n_hidden_h=10,
                 n_hidden_m=10, n_layers=2):
        super(PM1, self).__init__()

        self.n_hidden_h = n_hidden_h
        self.n_hidden_m = n_hidden_m
        self.sequence_len_h = sequence_len_h
        self.sequence_len_m = sequence_len_m
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.lstm_h = nn.LSTM(input_size=n_features_h,
                              hidden_size=n_hidden_h,
                              num_layers=n_layers,
                              batch_first=True,
                              bidirectional=True,
                              dropout=0)

        self.lstm_m = nn.LSTM(input_size=n_features_m,
                              hidden_size=n_hidden_m,
                              num_layers=n_layers,
                              batch_first=True,
                              bidirectional=True,
                              dropout=0)
        self.fcn_model = nn.Linear(in_features=n_features_m, out_features=10)
        # self.fcn_model =  nn.Linear(in_features=n_hidden_m*2 , out_features=20)
        self.fcn_model_2 = nn.Linear(in_features=10, out_features=5)
        self.fcn_h0 = nn.Linear(in_features=n_hidden_h * 2, out_features=10)
        self.fcn_hm = nn.Linear(in_features=5 + 10 + 1, out_features=10)
        self.fcn_h = nn.Linear(in_features=n_hidden_h * 2, out_features=10)
        self.fcn1 = nn.Linear(in_features=10, out_features=4)
        self.fcn2 = nn.Linear(in_features=4, out_features=1)
        # self.fcn3 = nn.Linear(in_features=2, out_features=1)

        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

    def reset_hidden_state_h(self):
        self.hidden_h = (
            torch.zeros(self.n_layers * 2, self.batch_size, self.n_hidden_h).to(device),
            torch.zeros(self.n_layers * 2, self.batch_size, self.n_hidden_h).to(device)
        )

    def reset_hidden_state_m(self):
        self.hidden_m = (
            torch.zeros(self.n_layers * 2, self.batch_size, self.n_hidden_m).to(device),
            torch.zeros(self.n_layers * 2, self.batch_size, self.n_hidden_m).to(device)
        )

    def forward(self, inputs_h, inputs_m, inputs_mask, y_max, y_min, mu_fix, sigma_fix, xi_p_fix, xi_n_fix, hetero):

        self.reset_hidden_state_h()
        self.reset_hidden_state_m()
        self.mu_fix = mu_fix
        self.sigma_fix = sigma_fix
        self.xi_p_fix = xi_p_fix
        self.xi_n_fix = xi_n_fix

        inputs_m_max = torch.amax(inputs_m, 1)
        inputs_m_max_mean = torch.mean(inputs_m_max, 1)

        inputs_mask_mean = torch.mean(inputs_mask, 1)
        if hetero != "m":
            lstm_out_h, self.hidden_h = self.lstm_h(inputs_h.view(self.batch_size, self.sequence_len_h, -1), self.hidden_h)
            out_h = lstm_out_h[:, -1, :]  # getting only the last time step's hidden state of the last layer
        else:
            lstm_out_m, self.hidden_m= self.lstm_m(inputs_m.view(self.batch_size, self.sequence_len_m, -1),  self.hidden_m)
            out = lstm_out_m[:, -1, :]  # getting only the last time step's hidden state of the last layer
        if hetero == "h+m":
            # if lstm in model data
            # lstm_out_m, self.hidden_m = self.lstm_m(inputs_m.view(self.batch_size, self.sequence_len_m, -1), self.hidden_m)  # lstm_out (batch_size, seq_len, hidden_size*2)
            # out_m = lstm_out_m[:, -1, :]
            # out_m = self.fcn_model(out_m)
            # if model max
            out_m = self.fcn_model(inputs_m_max.view(self.batch_size, -1))
            out_m = self.fcn_model_2(out_m)
            # out_m = inputs_mask.view(-1, 1).float()*out_m
            out_h = self.fcn_h0(out_h)
            out = torch.cat((out_h, out_m), 1)
            out = torch.cat((inputs_mask_mean.view(-1, 1).float(), out), 1)
            out = self.fcn_hm(out)
        elif hetero == "h":
            out = out_h
            out = self.fcn_h(
                out)  # feeding lstm output to a fully connected network which outputs 3 nodes: mu, sigma, xi

        out = self.fcn1(out)
        mu = out[:, 0] - self.mu_fix  # mu: first node of the fully connected network
        p1 = out[:, 1]  # sigma: second node of the fully connected network
        p2 = out[:, 2]
        p3 = out[:, 3]
        p2 = self.softplus(p2)
        p3 = self.softplus(p2)
        sigma = self.softplus(p1) - self.sigma_fix
        xi_p = ((sigma / (mu - y_min)) * (1 + boundary_tolerance) - (p2)) - self.xi_p_fix
        xi_n = ((p3) - (sigma / (y_max - mu)) * (1 + boundary_tolerance)) - self.xi_n_fix
        xi_p[xi_p > 0.95] = torch.tensor(0.95)
        yhat = self.fcn2(out)
        # yhat = self.fcn3(out)

        return mu, sigma, xi_p, xi_n, yhat

