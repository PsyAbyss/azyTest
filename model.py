import numpy as np
import torch
from attn import FullAttention, AttentionLayer

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

class NASDataTheftDetection(torch.nn.Module):
    def __init__(self, user_num, item_num, args, prob_dropout=0.1):
        super(NASDataTheftDetection, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.time_interval_emb = torch.nn.Embedding(args.time_span + 1, args.hidden_units)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        self.prob_dropout = torch.nn.Dropout(prob_dropout)
        # Attn = CIFAttention if args.attn == 'cifa' else FullAttention
        Attn = FullAttention
        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            # new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
            #                                                 args.num_heads,
            #                                                 args.dropout_rate)

            new_attn_layer = AttentionLayer(Attn(True,  attention_dropout=args.dropout_rate,
                                                 output_attention=args.output_attention,),
                                            args.hidden_units, args.num_heads, mix=False)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(self, log_seqs, time_intervals, time_interval_emb):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev)) # batch_size * max_len * hidden_size
        seqs *= self.item_emb.embedding_dim ** 0.5 #所有数值乘以hidden_size的开方
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1]) #初始化位置矩阵shape(128, 200)，每个元素为0-maxlen的list
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))

        if time_interval_emb:
            seqs += self.time_interval_emb(torch.LongTensor(time_intervals).to(self.dev))

        seqs = self.emb_dropout(seqs)
        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev) # torch.Size([128, 200]) 为0为True
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim 即原值为0的位置变回长度为hidden_size的全0向量

        # tl = seqs.shape[1] # time dim len for enforce causality
        # attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev)) #maxlen*maxlen矩阵，右上三角为True其他为False

        for i in range(len(self.attention_layers)):
            Q = self.attention_layernorms[i](seqs) # seq的LayerNorm结果: [B L D]
            mha_outputs, A = self.attention_layers[i](Q, seqs, seqs, attn_mask=None, key_padding_mask=timeline_mask)
            seqs = Q + self.prob_dropout(mha_outputs) #[B L D]
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)  # 即原值为0的位置变回长度为hidden_size的全0向量

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)  batch_size * max_len * hidden_size

        return log_feats, A

    def forward(self, user_ids, log_seqs, time_intervals, time_interval_emb, pos_seqs, neg_seqs): # for training
        log_feats, _ = self.log2feats(log_seqs, time_intervals, time_interval_emb) # user_ids hasn't been used yet  batch_size * max_len * hidden_size

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1) # 正样本 batch_size * max_len
        neg_logits = (log_feats * neg_embs).sum(dim=-1) # 负样本 batch_size * max_len

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, time_intervals, time_interval_emb, item_indices): # for inference
        log_feats, A = self.log2feats(log_seqs, time_intervals, time_interval_emb) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)
