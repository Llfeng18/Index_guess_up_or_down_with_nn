import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import math

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        # eps是为了防止归一化时候除以方差平方开根号时候, 方差为0时候出现错误,  $z_{\text{norm}}^{(i)} = \frac{z^{(i)} - \mu}{\sqrt{\sigma^2 + \epsilon}}$
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layers - 1)])
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.pos = nn.Parameter(torch.arange(0, input_size, dtype=torch.float).unsqueeze(0) * 100)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # 输入层
        x = x + self.pos.to(x.device)
        x = self.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        # 输出层
        x = self.output_layer(x)
        return x

class MLP_Residual(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers):
        super(MLP_Residual, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        # self.ln_layers = nn.ModuleList([nn.BatchNorm1d(hidden_size) for _ in range(num_hidden_layers)])
        # self.ln_layers = nn.ModuleList([LayerNorm(hidden_size, True) for _ in range(num_hidden_layers)])
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layers - 1)])
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # 输入层
        x = self.relu(self.input_layer(x))
        # 隐藏层带有残差连接
        for i in range(len(self.hidden_layers)):
            residual = x
            # x = self.relu(self.ln_layers[i](self.hidden_layers[i](x)))
            x = self.relu(self.hidden_layers[i](x))
            x = x + residual  # 残差连接，避免使用原地操作
        # 输出层
        # x = self.ln_layers[len(self.ln_layers)-1](x)
        x = self.output_layer(x)
        return x


class MLP_Tanh(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers):
        super(MLP_Tanh, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layers - 1)])
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, nonlinearity='tanh')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # 输入层
        x = self.tanh(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.tanh(layer(x))
        # 输出层
        x = self.output_layer(x)
        return x


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        input_size = 1
        # 为每一层RNN定义权重矩阵
        self.input_weights = nn.ModuleList(
            [nn.Linear(input_size, hidden_size) if i == 0 else nn.Linear(hidden_size, hidden_size) for i in
             range(num_layers)])
        self.hidden_weights = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])

        # 输出层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h = [torch.zeros(x.size(0), self.hidden_size).to(x.device) for _ in range(self.num_layers)]

        for t in range(x.size(1)):
            for l in range(self.num_layers):
                input = x[:, t, np.newaxis] if l == 0 else h[l - 1]
                h[l] = torch.tanh(self.input_weights[l](input) + self.hidden_weights[l](h[l]))

        out = self.fc(h[-1])
        return out

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        input_size = 1
        # 为每一层GRU定义权重矩阵
        self.W_ir = nn.ModuleList(
            [nn.Linear(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])
        self.W_hr = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.W_iz = nn.ModuleList(
            [nn.Linear(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])
        self.W_hz = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.W_in = nn.ModuleList(
            [nn.Linear(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])
        self.W_hn = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])

        # 输出层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h = [torch.zeros(x.size(0), self.hidden_size).to(x.device) for _ in range(self.num_layers)]

        for t in range(x.size(1)):
            for l in range(self.num_layers):
                input = x[:, t, np.newaxis] if l == 0 else h[l - 1]

                r = torch.sigmoid(self.W_ir[l](input) + self.W_hr[l](h[l]))
                z = torch.sigmoid(self.W_iz[l](input) + self.W_hz[l](h[l]))
                n = torch.tanh(self.W_in[l](input) + r * self.W_hn[l](h[l]))
                h[l] = (1 - z) * n + z * h[l]

        out = self.fc(h[-1])
        return out


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        input_size = 1
        # 为每一层LSTM定义权重矩阵
        self.W_ii = nn.ModuleList(
            [nn.Linear(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])
        self.W_hi = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.W_if = nn.ModuleList(
            [nn.Linear(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])
        self.W_hf = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.W_ig = nn.ModuleList(
            [nn.Linear(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])
        self.W_hg = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.W_io = nn.ModuleList(
            [nn.Linear(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])
        self.W_ho = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])

        # 输出层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h = [torch.zeros(x.size(0), self.hidden_size).to(x.device) for _ in range(self.num_layers)]
        c = [torch.zeros(x.size(0), self.hidden_size).to(x.device) for _ in range(self.num_layers)]

        for t in range(x.size(1)):
            for l in range(self.num_layers):
                input = x[:, t, np.newaxis] if l == 0 else h[l - 1]

                i = torch.sigmoid(self.W_ii[l](input) + self.W_hi[l](h[l]))
                f = torch.sigmoid(self.W_if[l](input) + self.W_hf[l](h[l]))
                g = torch.tanh(self.W_ig[l](input) + self.W_hg[l](h[l]))
                o = torch.sigmoid(self.W_io[l](input) + self.W_ho[l](h[l]))

                c[l] = f * c[l] + i * g
                h[l] = o * torch.tanh(c[l])

        out = self.fc(h[-1])
        return out


class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_size, n_head):
        super().__init__()
        assert hidden_size % n_head == 0
        self.c_attn = nn.Linear(hidden_size, 3 * hidden_size)
        self.c_proj = nn.Linear(hidden_size, hidden_size)
        self.n_head = n_head
        self.n_embd = hidden_size
        # self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        self.flash = False  # 当前pytorch不支持, 手动关闭

    def forward(self, x):
        B, T, C = x.size()

        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = F.softmax(att, dim=-1)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.c_proj(y)
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, hidden_size, n_head):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.attn = CausalSelfAttention(hidden_size, n_head)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(hidden_size, 4 * hidden_size),
            c_proj  = nn.Linear(4 * hidden_size, hidden_size),
            act     = nn.Sigmoid(),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(m.act(m.c_fc(x))) # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

class Transformer(nn.Module):
    """ Transformer Language Model, exactly as seen in GPT-2 """

    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers, n_head):
        super().__init__()
        self.block_size = input_size

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Linear(1, hidden_size),
            # wte = nn.Embedding(input_size, hidden_size),
            wpe = nn.Embedding(input_size, hidden_size),
            h = nn.ModuleList([Block(hidden_size, n_head) for _ in range(num_hidden_layers)]),
            ln_f = nn.LayerNorm(hidden_size),
        ))
        self.lm_head = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, x):
        device = x.device
        b, t = x.size()
        pos = torch.arange(0, self.block_size, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, t)
        # print(x.shape)
        tok_emb = self.transformer.wte(x.view(b, t, 1))  # token embeddings of shape (b, t, n_embd)
        # print(tok_emb.shape)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (1, t, n_embd)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        # print(logits.shape)
        return logits[:, -1, :] .view(-1, logits.size(-1))


def init_model(model_type, input_size, hidden_size, output_size, num_hidden_layers, n_head, device):
    if model_type == "MLP_Tanh":
        model = MLP_Tanh(input_size, hidden_size, output_size, num_hidden_layers).to(device)
    elif model_type == "MLP_Residual":
        model = MLP_Residual(input_size, hidden_size, output_size, num_hidden_layers).to(device)
    elif model_type == "MLP":
        model = MLP(input_size, hidden_size, output_size, num_hidden_layers).to(device)
    elif model_type == "RNN":
        model = RNN(input_size, hidden_size, output_size, num_hidden_layers).to(device)
    elif model_type == "GRU":
        model = GRU(input_size, hidden_size, output_size, num_hidden_layers).to(device)
    elif model_type == "LSTM":
        model = LSTM(input_size, hidden_size, output_size, num_hidden_layers).to(device)
    elif model_type == "Transformer":
        model = Transformer(input_size, hidden_size, output_size, num_hidden_layers, n_head).to(device)
    else:
        print(model_type)
        model = None

    return model