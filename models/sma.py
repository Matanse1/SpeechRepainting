import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# This code was adapated from https://github.com/keonlee9420/Stepwise_Monotonic_Multihead_Attention/tree/main with some modifiction for my purpose

"""
You may change these hyperparameters depending on the task.
"""

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
# If True, the stepwise monotonice multihead attention is activated. Else, it is a normal multihead attention just like in Transformer.
hp = AttrDict({"sma_head": 3, "sma_dropout": 0.1, "sma_tunable": True})



class StepwiseMonotonicMultiheadAttention(nn.Module):
    """ Stepwise Monotonic Multihead Attention
    args:
        n_heads (int): number of monotonic attention heads
        d_model (int): dimension of model (attention)
        d_k (int): dimension of key
        d_v (int): dimension of value
        noise_std (float): standard deviation for input noisse
        dropout (float): dropout probability for attention weights
    """

    def __init__(self, d_model, d_k, d_v, 
                                    noise_std=1.0, 
                                    n_head=hp.sma_head,
                                    dropout=hp.sma_dropout, 
                                    is_tunable=hp.sma_tunable):
        super(StepwiseMonotonicMultiheadAttention, self).__init__()
        self.n_head = n_head
        self.noise_std = noise_std
        self.energy = MultiheadEnergy(n_head, d_model, d_k, d_v)

        self.dropout = nn.Dropout(dropout)
        self.last_layer = nn.Linear(n_head*d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

        self.is_tunable = is_tunable

    def add_gaussian_noise(self, xs, std):
        """Add Gaussian noise to encourage discreteness."""
        noise = xs.new_zeros(xs.size()).normal_(std=std)
        return xs + noise

    def expectation(self, e, aw_prev, n_head):
        """
        e --- [batch*n_head, qlen, klen]
        aw_prev --- [batch*n_head, qlen, 1]
        See https://gist.github.com/mutiann/38a7638f75c21479582d7391490df37c
        See https://github.com/hirofumi0810/neural_sp/blob/093bfade110d5a15a4f7a58fffe8d235acbfe14f/neural_sp/models/modules/mocha.py#L430
        """
        batch_size, qlen, klen = aw_prev.size(0)//n_head, e.size(1), e.size(2)

        # Compute probability sampling matrix P
        p_sample = torch.sigmoid(self.add_gaussian_noise(e, self.noise_std) if self.training else e)  # [batch*n_head, qlen, klen]

        alpha = []
        # Compute recurrence relation solution along mel frame domain
        for i in range(klen):
            p_sample_i = p_sample[:, :, i:i + 1]
            pad = torch.zeros([batch_size*n_head, 1, 1], dtype=aw_prev.dtype).to(aw_prev.device)
            aw_prev = aw_prev * p_sample_i + torch.cat(
                    (pad, aw_prev[:, :-1, :] * (1.0 - p_sample_i[:, :-1, :])), dim=1)
            alpha.append(aw_prev)

        alpha = torch.cat(alpha, dim=-1) if klen > 1 else alpha[-1] # [batch*n_head, qlen, klen]

        assert not torch.isnan(alpha).any(), "NaN detected in alpha."

        return alpha, p_sample

    def focused_head(self, multihead, mel_len):
        """
        Apply focus rate to select the best diagonal head.
        multihead --- [batch*n_heads, seq_len, mel_len]
        mel_len --- [batch,]
        return --- [batch, seq_len, mel_len]
        """
        # [batch*n_heads, seq_len, mel_len] -> [batch, n_heads, seq_len, mel_len]
        multihead = multihead.reshape(self.n_head, -1, multihead.size(1), multihead.size(2)).transpose(0, 1)
        focus_rate = torch.max(multihead, dim=2)[0].sum(dim=-1)/(mel_len.unsqueeze(1)) # [batch, n_heads]
        h_idx = torch.argmax(focus_rate, dim=1) # [batch,]
        batch=list()
        fr_max=0
        for b, fr, i in zip(multihead, focus_rate, h_idx):
            batch.append(b[i])
            fr_max += fr[i].detach().item()
        return torch.stack(batch), fr_max/h_idx.size(0)
    
    def repeat_mask_multihead(self, mask):
        """
        Repeat mask over multihead.
        mask --- [batch, qlen, klen]
        return --- [batch*n_head, qlen, klen]
        """
        return mask.repeat(self.n_head, 1, 1)

    def forward(self, q, k, v, mel_len, q_mask=None, k_mask=None, aw_prev=None):
        batch_size, qlen, klen = q.size(0), q.size(1), k.size(1)
        
        if q_mask is not None and k_mask is not None:
            attn_mask = q_mask.unsqueeze(2) | k_mask.unsqueeze(1)  # [B, T1, T2]
            attn_mask = attn_mask.unsqueeze(1).expand(-1, self.n_head, -1, -1)  # [B, n_head, T1, T2]
            attn_mask = attn_mask.permute(1, 0, 2, 3).contiguous().view(-1, qlen, klen)  # Flatten batch & heads: [(B*n_head), T1, T2]
        else:
            attn_mask = None
        # if mask is not None:
        #     mask = self.repeat_mask_multihead(mask)

        # Calculate energy
        e, v, all_inf_mask = self.energy(q, k, v, attn_mask)  # [batch*n_head, qlen, klen], [batch*n_head, klen, d_v]

        # Get alpha
        alpha_cv = F.softmax(e, dim=-1) # [batch*n_head, qlen, klen]

        # Masking to ignore padding (query side)
        if all_inf_mask is not None:
            alpha_cv = alpha_cv.masked_fill(all_inf_mask, 0)
            # query_mask = self.repeat_mask_multihead(query_mask.repeat(1, 1, klen))
            # alpha_cv = alpha_cv.masked_fill(query_mask, 0.)

        # Get focused alpha
        alpha_fc, fr_max = self.focused_head(alpha_cv, mel_len) # [batch, qlen, klen]

        if self.is_tunable:
            # Monotonic enhancement
            if aw_prev is None:
                aw_prev = k.new_zeros(batch_size, qlen, 1) # [batch, qlen, 1]
                aw_prev[:, 0:1] = k.new_ones(batch_size, 1, 1) # initialize with [1, 0, 0 ... 0]
            alpha_me, _ = self.expectation(alpha_fc, aw_prev, 1) # [batch, qlen, klen]

            # Calculate context vector
            v = v.reshape(self.n_head, batch_size, klen, -1).permute(1, 2, 0, 3) # [batch, klen, n_head, d_v]
            cv = torch.bmm(alpha_me, v.reshape(batch_size, klen, -1)) # [batch, qlen, n_head*d_v]
        else:
            # Calculate normal multihead attention
            cv = torch.bmm(alpha_cv, v).reshape(self.n_head, batch_size, qlen, -1).permute(1, 2, 0, 3) # [batch, qlen, n_head, d_v]
            cv = cv.reshape(batch_size, qlen, -1) # [batch, qlen, n_head*d_v]

        cv = self.dropout(self.last_layer(cv))
        # cv = self.layer_norm(cv) # maybe like the custon attention to de output = self.dropout(output) + residual without layer norm
        return cv, alpha_fc, fr_max


class MultiheadEnergy(nn.Module):
    """ Energy function for the (monotonic) multihead attention """

    def __init__(self, n_head, d_model, d_k, d_v):
        super(MultiheadEnergy, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        
        kernel_size = 13
        self.w_qs = nn.Conv1d(d_model, n_head * d_k, kernel_size, padding=kernel_size // 2, groups=n_head)
        self.w_ks = nn.Conv1d(d_model, n_head * d_k, kernel_size, padding=kernel_size // 2, groups=n_head)
        self.w_vs = nn.Conv1d(d_model, n_head * d_v, kernel_size, padding=kernel_size // 2, groups=n_head)
        
        # self.w_qs = nn.Linear(d_model, n_head * d_k)
        # self.w_ks = nn.Linear(d_model, n_head * d_k)
        # self.w_vs = nn.Linear(d_model, n_head * d_v)

        self.temperature = np.power(d_k, 0.5)
        self._init_weights()
        
    def _init_weights(self):
        for layer in [self.w_qs, self.w_ks, self.w_vs]:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
                
    def scaled_dot_product(self, q, k):
        sdp = torch.bmm(q, k.transpose(1, 2)) # (n*b) x lq x lk
        sdp = sdp / self.temperature
        return sdp

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        # q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        # k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        # v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        
        # Transpose to match Conv1d input format (batch, channels, sequence_length)
        q = q.permute(0, 2, 1)  # (batch, d_model, len_q)
        k = k.permute(0, 2, 1)  # (batch, d_model, len_k)
        v = v.permute(0, 2, 1)  # (batch, d_model, len_k)
        
        q = self.w_qs(q).permute(0, 2, 1).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).permute(0, 2, 1).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).permute(0, 2, 1).view(sz_b, len_k, n_head, d_v)
        
        q = q.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_v, d_v)  # (n*b) x lv x dv

        # Compute monotonic multihead energy
        e = self.scaled_dot_product(q, k) # (n*b) x lq x lk

        # Masking to ignore padding
        if mask is not None:
            e = e.masked_fill(mask, -np.inf) # maybe float(np.finfo(torch.tensor(0, dtype=e.dtype).numpy().dtype).min)
                        # Ensure no row is entirely masked (NaN issue)
            all_inf_rows = e == -np.inf
            all_inf_mask = all_inf_rows.all(dim=-1, keepdim=True)  # (B, T1, 1)
        else:
            all_inf_mask = None
            
        # if mask is not None:
        #     NEG_INF = float(np.finfo(torch.tensor(0, dtype=e.dtype).numpy().dtype).min)
        #     e = e.masked_fill(mask, NEG_INF)

        return e, v, all_inf_mask
