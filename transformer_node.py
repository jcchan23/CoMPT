import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import xavier_normal_small_init_, xavier_uniform_small_init_


# Model definition

def make_model(d_atom, d_edge, N=2, d_model=128, h=8, dropout=0.1, attenuation_lambda=0.1, max_length=100,
               N_dense=2, leaky_relu_slope=0.0, dense_output_nonlinearity='relu', distance_matrix_kernel='softmax',
               n_output=1, scale_norm=True, init_type='uniform', n_generator_layers=1,
               aggregation_type='mean'):
    """Helper: Construct a model from hyper-parameters."""
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model, d_atom, d_edge, leaky_relu_slope, dropout, attenuation_lambda, distance_matrix_kernel)
    ff = PositionwiseFeedForward(d_model, N_dense, dropout, leaky_relu_slope, dense_output_nonlinearity)
    model = GraphTransformer(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout, scale_norm), N, scale_norm),
        Node_Embeddings(d_atom, d_model, dropout),
        Edge_Embeddings(d_edge, d_model, dropout),
        Position_Encoding(max_length, d_model, dropout),
        Generator(d_model, n_output, n_generator_layers, leaky_relu_slope, dropout, scale_norm, aggregation_type)
    )

    # This was important from their code. Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            if init_type == 'uniform':
                nn.init.xavier_uniform_(p)
            elif init_type == 'normal':
                nn.init.xavier_normal_(p)
            elif init_type == 'small_normal_init':
                xavier_normal_small_init_(p)
            elif init_type == 'small_uniform_init':
                xavier_uniform_small_init_(p)
    return model


class GraphTransformer(nn.Module):
    def __init__(self, encoder, node_embed, edge_embed, pos_embed, generator):
        super(GraphTransformer, self).__init__()
        self.encoder = encoder
        self.node_embed = node_embed
        self.edge_embed = edge_embed
        self.pos_embed = pos_embed
        self.generator = generator

    def forward(self, src, src_mask, adj_matrix, edges_att):
        """Take in and process masked src and target sequences."""
        return self.predict(self.encode(src, src_mask, adj_matrix, edges_att), src_mask)

    def encode(self, src, src_mask, adj_matrix, edges_att):
        # src.shape = (batch, max_length, d_atom + 1)
        src_emb = self.node_embed(src[:, :, :-1]) + self.pos_embed(src[:, :, -1].squeeze(-1).long())
        # edges_att = src_emb.unsqueeze(-2) + self.edge_embed(edges_att)
        edges_att = self.edge_embed(edges_att)
        return self.encoder(src_emb, src_mask, adj_matrix, edges_att)

    def predict(self, out, out_mask):
        return self.generator(out, out_mask)

# Embeddings


class Node_Embeddings(nn.Module):
    def __init__(self, d_atom, d_emb, dropout):
        super(Node_Embeddings, self).__init__()
        self.lut = nn.Linear(d_atom, d_emb)
        self.dropout = nn.Dropout(dropout)
        self.d_emb = d_emb

    def forward(self, x):   # x.shape(batch, max_length, d_atom)
        return self.dropout(self.lut(x)) * math.sqrt(self.d_emb)


class Edge_Embeddings(nn.Module):
    def __init__(self, d_edge, d_emb, dropout):
        super(Edge_Embeddings, self).__init__()
        self.lut = nn.Linear(d_edge, d_emb)
        self.dropout = nn.Dropout(dropout)
        self.d_emb = d_emb

    def forward(self, x):  # x.shape = (batch, max_length, max_length, d_edge)
        return self.dropout(self.lut(x)) * math.sqrt(self.d_emb)


class Position_Encoding(nn.Module):
    def __init__(self, max_length, d_emb, dropout):
        super(Position_Encoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Embedding(max_length + 1, d_emb, padding_idx=0)

    def forward(self, x):
        return self.dropout(self.pe(x))   # (batch, max_length) -> (batch, max_length, d_emb)

# Generator
class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


def swish_function(x):
    return x * torch.sigmoid(x)


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


def mish_function(x):
    return x * torch.tanh(F.softplus(x))


class Generator(nn.Module):
    """Define standard linear + softmax generation step."""

    def __init__(self, d_model, n_output=1, n_layers=1,
                 leaky_relu_slope=0.01, dropout=0.0, scale_norm=False, aggregation_type='mean'):
        super(Generator, self).__init__()
        if n_layers == 1:
            self.proj = nn.Linear(d_model, n_output)
        else:
            self.proj = []
            for i in range(n_layers - 1):
                self.proj.append(nn.Linear(d_model, d_model))
                # self.proj.append(nn.LeakyReLU(leaky_relu_slope))
                self.proj.append(Mish())
                self.proj.append(ScaleNorm(d_model) if scale_norm else LayerNorm(d_model))
                self.proj.append(nn.Dropout(dropout))
            self.proj.append(nn.Linear(d_model, n_output))
            self.proj = torch.nn.Sequential(*self.proj)

        self.aggregation_type = aggregation_type
        self.leaky_relu_slope = leaky_relu_slope

        if self.aggregation_type == 'gru':
            self.gru = nn.GRU(d_model, d_model, batch_first=True, bidirectional=True)
            self.linear = nn.Linear(2 * d_model, d_model)
            self.bias = nn.Parameter(torch.Tensor(d_model))
            self.bias.data.uniform_(-1.0 / math.sqrt(d_model), 1.0 / math.sqrt(d_model))

    def forward(self, x, mask):
        mask = mask.unsqueeze(-1).float()
        out_masked = x * mask               # (batch, max_length, d_model)

        if self.aggregation_type == 'mean':
            out_sum = out_masked.sum(dim=1)
            mask_sum = mask.sum(dim=1)
            out_pooling = out_sum / mask_sum
        elif self.aggregation_type == 'sum':
            out_sum = out_masked.sum(dim=1)
            out_pooling = out_sum
        elif self.aggregation_type == 'summax':
            out_sum = torch.sum(out_masked, dim=1)
            out_max = torch.max(out_masked, dim=1)[0]
            out_pooling = out_sum * out_max
        elif self.aggregation_type == 'gru':
            # (batch, max_length, d_model)
            out_hidden = mish_function(out_masked + self.bias)
            out_hidden = torch.max(out_hidden, dim=1)[0].unsqueeze(0)       # (1, batch, d_model)
            out_hidden = out_hidden.repeat(2, 1, 1)                         # (2, batch, d_model)

            cur_message, cur_hidden = self.gru(out_masked, out_hidden)      # message = (batch, max_length, 2 * d_model)
            cur_message = self.linear(cur_message)                          # (batch, max_length, d_model)

            out_sum = cur_message.sum(dim=1)                                # (batch, d_model)
            mask_sum = mask.sum(dim=1)
            out_pooling = out_sum / mask_sum                                # (batch, d_model)
        else:
            out_pooling = out_masked
        projected = self.proj(out_pooling)
        return projected

# Encoder


class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, layer, N, scale_norm):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = ScaleNorm(layer.size) if scale_norm else LayerNorm(layer.size)

    def forward(self, x, mask, adj_matrix, edges_att):
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x, mask, adj_matrix, edges_att)
        return self.norm(x)


class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)"""

    def __init__(self, size, self_attn, feed_forward, dropout, scale_norm):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn        # MultiHeadedAttention
        self.feed_forward = feed_forward  # Position wise FeedForward
        self.sublayer = clones(SublayerConnection(size, dropout, scale_norm), 2)
        self.size = size

    def forward(self, x, mask, adj_matrix, edges_att):
        """Follow Figure 1 (left) for connections."""
        # x.shape = (batch, max_length, d_atom)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, adj_matrix, edges_att, mask))
        return self.sublayer[1](x, self.feed_forward)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout, scale_norm):
        super(SublayerConnection, self).__init__()
        self.norm = ScaleNorm(size) if scale_norm else LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))


# Conv 1x1 aka Positionwise feed forward

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, N_dense, dropout=0.1, leaky_relu_slope=0.1, dense_output_nonlinearity='relu'):
        super(PositionwiseFeedForward, self).__init__()
        self.N_dense = N_dense
        self.linears = clones(nn.Linear(d_model, d_model), N_dense)
        self.dropout = clones(nn.Dropout(dropout), N_dense)
        self.leaky_relu_slope = leaky_relu_slope
        if dense_output_nonlinearity == 'relu':
            self.dense_output_nonlinearity = lambda x: F.leaky_relu(x, negative_slope=self.leaky_relu_slope)
        elif dense_output_nonlinearity == 'tanh':
            self.tanh = torch.nn.Tanh()
            self.dense_output_nonlinearity = lambda x: self.tanh(x)
        elif dense_output_nonlinearity == 'gelu':
            self.dense_output_nonlinearity = lambda x: F.gelu(x)
        elif dense_output_nonlinearity == 'none':
            self.dense_output_nonlinearity = lambda x: x
        elif dense_output_nonlinearity == 'swish':
            self.dense_output_nonlinearity = lambda x: x * torch.sigmoid(x)
        elif dense_output_nonlinearity == 'mish':
            self.dense_output_nonlinearity = lambda x: x * torch.tanh(F.softplus(x))

    def forward(self, x):
        if self.N_dense == 0:
            return x

        for i in range(len(self.linears) - 1):
            x = self.dropout[i](mish_function(self.linears[i](x)))

        return self.dropout[-1](self.dense_output_nonlinearity(self.linears[-1](x)))


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    """Construct a layernorm module (See citation for details)."""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ScaleNorm(nn.Module):
    """ScaleNorm"""
    "All g’s in SCALE NORM are initialized to sqrt(d)"

    def __init__(self, scale, eps=1e-5):
        super(ScaleNorm, self).__init__()
        self.scale = nn.Parameter(torch.tensor(math.sqrt(scale)))
        self.eps = eps

    def forward(self, x):
        norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x * norm


# Attention


def attention(query_edge, key_edge, value_edge, adj_matrix, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention'"""
    # query_edge.shape = (batch, h, max_length, d_e)
    # key_edge.shape = (batch, h, max_length, max_length, d_e)
    # out_edge_scores.shape = (batch, h, max_length, max_length)
    # in_edge_scores.shape = (batch, h, max_length, max_length)

    d_e = query_edge.size(-1)
    out_edge_scores = torch.einsum('bhmd,bhmnd->bhmn', query_edge, key_edge) / math.sqrt(d_e)
    in_edge_scores = torch.einsum('bhnd,bhmnd->bhnm', query_edge, key_edge) / math.sqrt(d_e)

    if mask is not None:
        mask = mask.unsqueeze(1).repeat(1, query_edge.shape[1], query_edge.shape[2], 1)
        out_edge_scores = out_edge_scores.masked_fill(mask == 0, -np.inf)
        in_edge_scores = in_edge_scores.masked_fill(mask == 0, -np.inf)

    out_edge_attn = F.softmax(out_edge_scores, dim=-1)
    in_edge_attn = F.softmax(in_edge_scores, dim=-1)
    diag_edge_attn = torch.diag_embed(torch.diagonal(out_edge_attn, dim1=-2, dim2=-1), dim1=-2, dim2=-1)

    p_weighted = out_edge_attn + in_edge_attn - diag_edge_attn

    # add the diffusion caused by distance
    p_weighted = p_weighted * adj_matrix.unsqueeze(1)

    if dropout is not None:
        p_weighted = dropout(p_weighted)

    # p_weighted.shape = (batch, h, max_length, max_length), value.shape = (batch, h, max_length, d_k)
    message_features = torch.matmul(p_weighted, value_edge)
    return message_features, p_weighted


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, d_atom, d_edge, leaky_relu_slope=0.1, dropout=0.1, attenuation_lambda=0.1,
                 distance_matrix_kernel='softmax'):
        """Take in model size and number of heads."""
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h  # We assume d_v always equals d_k
        self.h = h

        self.attenuation_lambda = torch.nn.Parameter(torch.tensor(attenuation_lambda, requires_grad=True))

        self.linears = clones(nn.Linear(d_model, d_model), 4)  # 4 for query, key, value and output

        self.attn = None
        self.leaky_relu_slope = leaky_relu_slope
        self.dropout = nn.Dropout(p=dropout)

        if distance_matrix_kernel == 'softmax':
            self.distance_matrix_kernel = lambda x: F.softmax(-x, dim=-1)
        elif distance_matrix_kernel == 'exp':
            self.distance_matrix_kernel = lambda x: torch.exp(-x)

    def forward(self, query, key, value, adj_matrix, edges_att, mask=None):

        """Implements Figure 2"""
        mask = mask.unsqueeze(1) if mask is not None else mask
        n_batches, max_length, d_model = query.shape

        # 1) Do all the linear projections in batch from d_model => h x d_k

        # Prepare adjacency matrix with shape (batch, max_length, max_length)
        torch.clamp(self.attenuation_lambda, min=0, max=1)
        adj_matrix = self.attenuation_lambda * adj_matrix
        adj_matrix = adj_matrix.masked_fill(mask.repeat(1, mask.shape[-1], 1) == 0, np.inf)
        adj_matrix = self.distance_matrix_kernel(adj_matrix)

        query_edge = self.linears[0](query).view(n_batches, max_length, self.h, self.d_k).transpose(1, 2)
        key_edge = self.linears[1](edges_att).view(n_batches, max_length, max_length, self.h, self.d_k).permute(0, 3, 1, 2, 4)
        value_edge = self.linears[2](value).view(n_batches, max_length, self.h, self.d_k).transpose(1, 2)

        # 2) Apply attention on all the projected vectors in batch.

        x, self.attn = attention(query_edge, key_edge, value_edge, adj_matrix, mask=mask, dropout=self.dropout)
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(n_batches, -1, self.h * self.d_k)
        return self.linears[-1](x)
