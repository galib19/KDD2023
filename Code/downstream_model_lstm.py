class LSTM_GEV(nn.Module):
    def __init__(self, n_features_h, n_features_m, sequence_len_h, sequence_len_m, batch_size=64, n_hidden_h=10,
                 n_hidden_m=5, n_layers=2):
        super(LSTM_GEV, self).__init__()

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
        self.fcn_model_2 = nn.Linear(in_features=10, out_features=5)
        self.fcn_h0 = nn.Linear(in_features=n_hidden_h * 2, out_features=10)
        self.fcn_hm = nn.Linear(in_features=5 + 10 + 1, out_features=10)
        self.fcn_h = nn.Linear(in_features=n_hidden_h * 2, out_features=10)
        self.fcn1 = nn.Linear(in_features=10, out_features=4)
        self.fcn2 = nn.Linear(in_features=4, out_features=1)

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

        inputs_m_max = torch.amax(inputs_m, 1)
        inputs_m_max_mean = torch.mean(inputs_m_max, 1)

        inputs_mask_mean = torch.mean(inputs_mask, 1)
        if hetero != "m":
            lstm_out_h, self.hidden_h = self.lstm_h(inputs_h.view(self.batch_size, self.sequence_len_h, -1),
                                                    self.hidden_h)
            out_h = lstm_out_h[:, -1, :]  # getting only the last time step's hidden state of the last layer
        if hetero == "m":
            lstm_out_m, self.hidden_h = self.lstm_m(inputs_m.view(self.batch_size, self.sequence_len_m, -1),
                                                    self.hidden_m)
            out_h = lstm_out_m[:, -1, :]
        elif hetero == "h+m":
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
        yhat = self.fcn2(out)
        # yhat = self.fcn3(out)
        mu = torch.tensor([0.0, 0.0])
        sigma = torch.tensor([0.0, 0.0])
        xi_p = torch.tensor([0.0, 0.0])
        xi_n = torch.tensor([0.0, 0.0])
        return mu, sigma, xi_p, xi_n, yhat


class FCN(nn.Module):
  # output: max
  # train_loss: max
  def __init__(self, n_features_h, n_features_m, sequence_len_h, sequence_len_m, batch_size=64, n_hidden_h=10,
               n_hidden_m=5, n_layers=2):
    super(FCN, self).__init__()

    self.n_hidden_h = n_hidden_h
    self.n_hidden_m = n_hidden_m
    self.sequence_len_h = sequence_len_h
    self.sequence_len_m = sequence_len_m
    self.n_layers = n_layers
    self.batch_size = batch_size

    self.linear1 = nn.Linear(in_features=sequence_len, out_features=n_hidden_h)
    self.linear1_m = nn.Linear(in_features=sequence_len*,n_features_m out_features=n_hidden_m)
    self.fcn_model = nn.Linear(in_features=n_features_m, out_features=10)
    self.fcn_model_2 = nn.Linear(in_features=10, out_features=5)
    self.fcn_h0 = nn.Linear(in_features=n_hidden_h * 2, out_features=10)
    self.fcn_hm = nn.Linear(in_features=5 + 10 + 1, out_features=10)
    self.linear2 = nn.Linear(in_features=n_hidden_h, out_features=n_hidden_h)
    self.linear3 = nn.Linear(in_features=n_hidden_h, out_features=n_hidden_h)
    self.linearFinal = nn.Linear(in_features=n_hidden_h, out_features=1)
    self.fcn_h = nn.Linear(in_features=n_hidden_h * 2, out_features=10)
    self.fcn1 = nn.Linear(in_features=10, out_features=4)
    self.fcn2 = nn.Linear(in_features=4, out_features=1)

    self.sigmoid = nn.Sigmoid()
    self.softplus = nn.Softplus()




  def forward(self, inputs_h, inputs_m, inputs_mask, y_max, y_min, mu_fix, sigma_fix, xi_p_fix, xi_n_fix, hetero):

      inputs_m_max = torch.amax(inputs_m, 1)
      inputs_m_max_mean = torch.mean(inputs_m_max, 1)

      inputs_mask_mean = torch.mean(inputs_mask, 1)
      if hetero != "m":
          out = self.linear1(inputs_h.view(self.batch_size, -1))
      if hetero == "m":
          out = self.linear1_m(inputs_m.view(self.batch_size, -1))
      elif hetero == "h+m":
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

      out = self.linear1(inputs_h.view(self.batch_size, -1))
      out = torch.relu(out)
      out = self.linear2(out)
      out = torch.relu(out)
      out = self.linear3(out)
      yhat = torch.relu(out)
      mu = torch.tensor([0.0, 0.0])
      sigma = torch.tensor([0.0, 0.0])
      xi_p = torch.tensor([0.0, 0.0])
      xi_n = torch.tensor([0.0, 0.0])
      return mu, sigma, xi_p, xi_n, yhat



class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransAm(nn.Module):
    def __init__(self, feature_size = 250, n_features_h, n_features_m, sequence_len_h, sequence_len_m, batch_size=64, n_hidden_h=10,
                 n_hidden_m=5, n_layers=2):
        super(TransAm, self).__init__()

        self.n_hidden_h = n_hidden_h
        self.n_hidden_m = n_hidden_m
        self.sequence_len_h = sequence_len_h
        self.sequence_len_m = sequence_len_m
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.fcn_model = nn.Linear(in_features=n_features_m, out_features=10)
        self.fcn_model_2 = nn.Linear(in_features=10, out_features=5)
        self.fcn_h0 = nn.Linear(in_features=n_hidden_h * 2, out_features=10)
        self.fcn_hm = nn.Linear(in_features=5 + 10 + 1, out_features=10)
        self.fcn_h = nn.Linear(in_features=n_hidden_h * 2, out_features=10)
        self.fcn1 = nn.Linear(in_features=10, out_features=4)
        self.fcn2 = nn.Linear(in_features=4, out_features=1)

        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

        self.model_type = 'Transformer'
        self.feature_size = feature_size
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=4, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder1 = nn.Linear(train_time_steps * feature_size, 50)
        self.decoder2 = nn.Linear(50, 10)
        self.decoder3 = nn.Linear(10, 4)
        self.decoder4 = nn.Linear(4, 10)
        self.decoder5 = nn.Linear(10, 1)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder1.bias.data.zero_()
        self.decoder1.weight.data.uniform_(-initrange, initrange)

    def forward(self, inputs_h, inputs_m, inputs_mask, y_max, y_min, mu_fix, sigma_fix, xi_p_fix, xi_n_fix, hetero):
        if hetero == "h":
            src_h = src_h.reshape(train_time_steps, batch_size, -1)
            src = self.pos_encoder(src_h)
        elif hetero == "m":
            src_m = src_m.reshape(train_time_steps, batch_size, -1)
            src = self.pos_encoder(src_m)
        elif hetero == "h+m":
            src = torch.cat((src_h, src_m), 1)
            src = torch.cat((inputs_mask_mean.view(-1, 1).float(), src), 1)
            src = self.fcn_hm(src)
            src = src.reshape(train_time_steps, batch_size, -1)
            src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)  # , self.src_mask)
        output = output.reshape(batch_size, train_time_steps * self.feature_size)
        output = self.decoder1(output)
        output = self.decoder2(output)
        output = self.decoder3(output)
        output = self.decoder4(output)
        output = self.decoder5(output)

        mu = torch.tensor([0.0, 0.0])
        sigma = torch.tensor([0.0, 0.0])
        xi_p = torch.tensor([0.0, 0.0])
        xi_n = torch.tensor([0.0, 0.0])
        return mu, sigma, xi_p, xi_n, output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class DeepPIPE(nn.Module):

    def __init__(self, n_features_h, n_features_m, sequence_len_h, sequence_len_m, batch_size=64, n_hidden_h=10,
                 n_hidden_m=5, n_layers=2):
        super(DeepPIPE, self).__init__()

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
        self.fcn_model_2 = nn.Linear(in_features=10, out_features=5)
        self.fcn_h0 = nn.Linear(in_features=n_hidden_h * 2, out_features=10)
        self.fcn_hm = nn.Linear(in_features=5 + 10 + 1, out_features=10)
        self.fcn_h = nn.Linear(in_features=n_hidden_h * 2, out_features=10)
        self.fcn1 = nn.Linear(in_features=10, out_features=4)
        self.fcn2 = nn.Linear(in_features=4, out_features=1)
        self.linear_y = nn.Linear(in_features=3, out_features=1)
        self.linear_p1 = nn.Linear(in_features=3, out_features=1)
        self.linear_p2 = nn.Linear(in_features=3, out_features=1)

        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

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

        inputs_m_max = torch.amax(inputs_m, 1)
        inputs_m_max_mean = torch.mean(inputs_m_max, 1)

        inputs_mask_mean = torch.mean(inputs_mask, 1)
        if hetero != "m":
            lstm_out_h, self.hidden_h = self.lstm_h(inputs_h.view(self.batch_size, self.sequence_len_h, -1),
                                                    self.hidden_h)
            out_h = lstm_out_h[:, -1, :]  # getting only the last time step's hidden state of the last layer
        if hetero == "m":
            lstm_out_m, self.hidden_h = self.lstm_m(inputs_m.view(self.batch_size, self.sequence_len_m, -1),
                                                    self.hidden_m)
            out_h = lstm_out_m[:, -1, :]
        elif hetero == "h+m":
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

        out = lstm_out[:, -1, :]  # getting only the last time step's hidden state of the last layer
        # print("hidden states mean, std, min, max: ", lstm_out[:,:,:].mean().item(), lstm_out[:,:,:].std().item(), lstm_out[:,:,:].min().item(), lstm_out[:,:,:].max().item()) # lstm_out.shape -> out.shape: 64,16,100 -> 64,16. Batch size: 64, input_seq_len:  16, n_hidden*2 = 50*2 = 100 // *2 for bidirectional lstm
        out = self.fcn(out)  # feeding lstm output to a fully connected network which outputs 3 nodes: mu, sigma, xi
        out = self.fcn2(out)
        out = self.fcn3(out)

        yhat = self.linear_y(out)
        p1 = self.linear_p1(out)
        p2 = self.linear_p2(out)
        p1 = self.softplus(p1)
        p2 = self.softplus(p2)

        mu = torch.tensor([0.0, 0.0])
        sigma = torch.tensor([0.0, 0.0])
        xi_p = torch.tensor([0.0, 0.0])
        xi_n = torch.tensor([0.0, 0.0])
        return mu, sigma, xi_p, xi_n, yhat, p1, p2
