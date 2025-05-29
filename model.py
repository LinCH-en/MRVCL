import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from collections import OrderedDict
from loss import contrastive_loss, compute_mi
from MI import logsumexp, log_density
from GCN import WGCN
from HE import HetEmb


class MRVCL(nn.Module):
    def __init__(self, embedding_dim, output_size, num_heads, dropout_rate, node_num, state_dim, relation_dim, hidden_units):
        super(MRVCL, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.node_embeddings = nn.Embedding(node_num, embedding_dim)
        self.graph_layer = WGCN(embedding_dim, output_size, num_heads, dropout_rate)
        self.context_dim = embedding_dim
        self.het_emb = HetEmb(self.context_dim, 48, 183)

        self.state_dim = state_dim
        self.relation_dim = relation_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_units

        self.feature_transform = nn.Linear(embedding_dim + self.context_dim * 2, self.embedding_dim)

        self.relation_prior_gru_layer1 = nn.GRUCell(self.relation_dim, self.hidden_dim)
        self.relation_prior_gru_layer2 = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.relation_prior_mean = nn.Linear(self.hidden_dim, self.relation_dim)
        self.relation_prior_logvar = nn.Linear(self.hidden_dim, self.relation_dim)

        self.state_mean = LinearUnit(self.hidden_dim * 2, self.state_dim, False)
        self.state_logvar = LinearUnit(self.hidden_dim * 2, self.state_dim, False)
        self.relation_mean = nn.Linear(self.hidden_dim, self.relation_dim)
        self.relation_logvar = nn.Linear(self.hidden_dim, self.relation_dim)

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=,
                                                                    dim_feedforward=, dropout=)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=)

        self.bi_rnn = nn.GRU(self.embedding_dim, self.hidden_dim, num_layers=, batch_first=True, dropout=, bidirectional=True)
        self.uni_rnn = nn.GRU(self.hidden_dim * 2, self.hidden_dim, num_layers=, batch_first=True, dropout=)

        self.mlp = nn.Linear(embedding_dim + self.context_dim * 2, 128)
        self.hidden_size = hidden_units
        self.attention_size = hidden_units
        self.query_layer = nn.Linear(128, self.hidden_size, bias=True)
        self.attention_layer = nn.Linear(self.hidden_size, self.attention_size, bias=True)
        self.value_layer = nn.Linear(self.hidden_size, 1)

        self.fc_zt = nn.Linear(self.relation_dim, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_dim * 3, node_num)
        self.dropout_layer = nn.Dropout(0.3)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=, weight_decay=)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=, patience=)

    def get_history_layer(self, x, ht):
        x = self.query_layer(x)
        v = torch.tanh(ht * self.attention_layer(x))
        vu = self.value_layer(v)
        alphas = F.softmax(vu, dim=1)
        output = torch.sum(x * alphas, dim=1)
        output = self.dropout_layer(output)
        return output

    def encode_and_sample_posterior(self, x):
        L = x.shape[1]

        x = x.permute(1, 0, 2)  # Convert to (seq_len, batch_size, feature_dim)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # Convert back to (batch_size, seq_len, feature_dim)

        features, _ = self.bi_rnn(x)
        backward = features[:, 0, self.hidden_dim:2 * self.hidden_dim]
        frontal = features[:, L - 1, :self.hidden_dim]
        gru_out_state = torch.cat((frontal, backward), dim=1)
        state_mean = self.state_mean(gru_out_state)
        state_logvar = self.state_logvar(gru_out_state)
        state_posterior = self.reparameterize(state_mean, state_logvar, random_sampling=True)

        relation_features, _ = self.uni_rnn(features)
        relation_mean = self.relation_mean(relation_features)
        relation_logvar = self.relation_logvar(relation_features)
        relation_posterior = self.reparameterize(relation_mean, relation_logvar, random_sampling=True)

        return state_mean, state_logvar, state_posterior, relation_mean, relation_logvar, relation_posterior, gru_out_state

    def calculate_loss(self, state_mean, state_logvar, state_posterior, relation_posterior_mean, relation_posterior_logvar, relation_posterior, relation_prior_mean, relation_prior_logvar, relation_prior, prediction, label):
        batch_size, n_frame, z_dim = relation_posterior_mean.size()
        mi_xs = compute_mi(state_posterior, (state_mean, state_logvar))
        mi_xzs = [compute_mi(relation_posterior_t, (relation_posterior_mean_t, relation_posterior_logvar_t)) for relation_posterior_t, relation_posterior_mean_t, relation_posterior_logvar_t in zip(relation_posterior.permute(1, 0, 2), relation_posterior_mean.permute(1, 0, 2), relation_posterior_logvar.permute(1, 0, 2))]
        mi_xz = torch.stack(mi_xzs).sum()

        prediction_loss = nn.CrossEntropyLoss()(prediction, label)

        state_mean = state_mean.view(-1, state_mean.shape[-1])
        state_logvar = state_logvar.view(-1, state_logvar.shape[-1])
        kld_state = -0.5 * torch.sum(1 + state_logvar - torch.pow(state_mean, 2) - torch.exp(state_logvar))

        relation_posterior_var = torch.exp(relation_posterior_logvar)
        relation_prior_var = torch.exp(relation_prior_logvar)
        kld_relation = 0.5 * torch.sum(relation_prior_logvar - relation_posterior_logvar + ((relation_posterior_var + torch.pow(relation_posterior_mean - relation_prior_mean, 2)) / relation_prior_var) - 1)
        kld_state, kld_relation = kld_state / batch_size, kld_relation / batch_size

        mutual_info_sr = torch.zeros(1).to(self.device)
        if True:
            _logq_s_tmp = log_density(state_posterior.unsqueeze(0).repeat(n_frame, 1, 1).view(n_frame, batch_size, 1, self.state_dim), state_mean.unsqueeze(0).repeat(n_frame, 1, 1).view(n_frame, 1, batch_size, self.state_dim), state_logvar.unsqueeze(0).repeat(n_frame, 1, 1).view(n_frame, 1, batch_size, self.state_dim))
            _logq_r_tmp = log_density(relation_posterior.transpose(0, 1).view(n_frame, batch_size, 1, z_dim), relation_posterior_mean.transpose(0, 1).view(n_frame, 1, batch_size, z_dim), relation_posterior_logvar.transpose(0, 1).view(n_frame, 1, batch_size, z_dim))
            _logq_sr_tmp = torch.cat((_logq_s_tmp, _logq_r_tmp), dim=3)
            logq_s = logsumexp(_logq_s_tmp.sum(3), dim=2, keepdim=False) - math.log(batch_size * 47255)
            logq_r = logsumexp(_logq_r_tmp.sum(3), dim=2, keepdim=False) - math.log(batch_size * 47255)
            logq_sr = logsumexp(_logq_sr_tmp.sum(3), dim=2, keepdim=False) - math.log(batch_size * 47255)
            mutual_info_sr = F.relu(logq_sr - logq_s - logq_r).mean()

        loss = prediction_loss + 0.05 * (kld_state + kld_relation + mutual_info_sr)
        return loss

    def forward(self, graphs, batch_x, label, history_x, batch_time, history_time, batch_cat, history_cat, poi_time, poi_cat, pos_encoding):
        L = batch_x.shape[1]
        h_length = history_x.shape[1]

        node_embeddings = self.node_embeddings(graphs.x.to(self.device))
        node_embeddings = self.graph_layer(node_embeddings, graphs.edge_index.to(self.device), graphs.edge_attr.to(self.device))
        embedding = nn.Embedding.from_pretrained(node_embeddings)
        node_embeddings = embedding(batch_x)
        history_embeddings = embedding(history_x)

        pt_emb, pc_emb, time_emb, cat_emb = self.het_emb(poi_time, poi_cat)
        pt_emb = nn.Embedding.from_pretrained(pt_emb)
        pc_emb = nn.Embedding.from_pretrained(pc_emb)
        time_emb = nn.Embedding.from_pretrained(time_emb)
        time_embeddings = time_emb(batch_time)
        history_time_embeddings = time_emb(history_time)
        cat_emb = nn.Embedding.from_pretrained(cat_emb)
        category_embeddings = cat_emb(batch_cat)
        history_category_embeddings = cat_emb(history_cat)

        node_embeddings = torch.cat((node_embeddings, time_embeddings, category_embeddings), dim=2)
        node_embeddings = self.feature_transform(node_embeddings)

        state_mean, state_logvar, state_posterior, relation_mean_posterior, relation_logvar_posterior, relation_posterior, out = self.encode_and_sample_posterior(node_embeddings)

        relation_mean_prior, relation_logvar_prior, relation_prior = self.sample_relation_prior_train(relation_posterior, random_sampling=self.training)

        z_flatten = relation_posterior.view(-1, relation_posterior.shape[2])
        state_expand = state_posterior.unsqueeze(1).expand(-1, L, self.state_dim)
        zf = torch.cat((relation_posterior, state_expand), dim=2)
        ht = relation_posterior[:, -1, :].unsqueeze(1)
        softplus = nn.Softplus()
        ht = softplus(self.fc_zt(ht))

        history_embeddings = torch.cat((history_embeddings, history_time_embeddings, history_category_embeddings), dim=2)
        history_embeddings = self.mlp(history_embeddings)
        history_embeddings *= math.sqrt(128)
        history_embeddings += pos_encoding[:, :h_length, :]

        history_out = self.get_history_layer(history_embeddings, ht)
        out = torch.cat((out, history_out), dim=-1)
        prediction = self.output_layer(out).squeeze(1)
        loss = self.calculate_loss(state_mean, state_logvar, state_posterior, relation_mean_posterior, relation_logvar_posterior, relation_posterior, relation_mean_prior, relation_logvar_prior, relation_prior, prediction, label)
        prediction = F.softmax(prediction, dim=-1)

        return prediction, loss

    def reparameterize(self, mean, logvar, random_sampling=True):
        if random_sampling:
            eps = torch.randn_like(logvar)
            std = torch.exp(0.5 * logvar)
            z = mean + eps * std
            return z
        return mean

    def sample_relation_prior_train(self, relation_posterior, random_sampling=True):
        batch_size, L = relation_posterior.shape[0], relation_posterior.shape[1]

        r_t = torch.zeros(batch_size, self.relation_dim).to(self.device)
        h_t_ly1 = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        h_t_ly2 = torch.zeros(batch_size, self.hidden_dim).to(self.device)

        relation_out = []
        relation_means = []
        relation_logvars = []

        for i in range(L):
            h_t_ly1 = self.relation_prior_gru_layer1(r_t, h_t_ly1)
            h_t_ly2 = self.relation_prior_gru_layer2(h_t_ly1, h_t_ly2)

            relation_mean_t = self.relation_prior_mean(h_t_ly2)
            relation_logvar_t = self.relation_prior_logvar(h_t_ly2)
            relation_prior = self.reparameterize(relation_mean_t, relation_logvar_t, random_sampling)

            relation_out.append(relation_prior.unsqueeze(1))
            relation_means.append(relation_mean_t.unsqueeze(1))
            relation_logvars.append(relation_logvar_t.unsqueeze(1))

            r_t = relation_posterior[:, i, :]

        relation_out = torch.cat(relation_out, dim=1)
        relation_means = torch.cat(relation_means, dim=1)
        relation_logvars = torch.cat(relation_logvars, dim=1)

        return relation_means, relation_logvars, relation_out