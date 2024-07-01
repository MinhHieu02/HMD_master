""" Hierarchical attention modules """
import torch
import torch.nn as nn

from onmt.utils.misc import aeq, sequence_mask, sequence_mask_herd
import math
import torch.nn.functional as F

class HierarchicalAttention(nn.Module):
    """Dynamic attention"""
    def __init__(self, gpu, dim, attn_type="general"):
        super(HierarchicalAttention, self).__init__()
 
        device = torch.device("cuda" if gpu == True else "cpu")
        self.device = device
        self.dim = dim
        self.attn_type = attn_type
        assert (self.attn_type in ["dot", "general", "mlp"]), (
            "Please select a valid attention type.")

        # Hierarchical attention
        if self.attn_type == "general":
            self.word_linear_in1 = nn.Linear(dim, dim, bias=False)
            self.word_linear_in2 = nn.Linear(dim, dim, bias=False)
            self.word_linear_in3 = nn.Linear(dim, dim, bias=False)
            self.sent_linear_in1 = nn.Linear(dim, dim, bias=False)
            self.sent_linear_in2 = nn.Linear(dim, dim, bias=False)
            self.sent_linear_in3 = nn.Linear(dim, dim, bias=False)
        else:
            raise NotImplementedError

        # mlp wants it with bias
        out_bias = self.attn_type == "mlp"
        self.linear_out = nn.Linear(dim * 2, dim, bias=out_bias)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()    

    def score(self, h_s, type, dist_num, h_t1, h_t2=None, h_t3=None):
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_dim = h_t1.size()
            
        h_s_ = h_s.transpose(1, 2)
        dist_lembda1 = 0.5
        dist_lembda2 = 0.5
        
        if type == 'word'and dist_num == "first":
            h_t1_ = self.word_linear_in1(h_t1)
            h_t1 = h_t1_.view(tgt_batch, 1, tgt_dim)
            return torch.bmm(h_t1, h_s_)
        
        elif type == 'word' and dist_num == "second": 
            h_t1_ = self.word_linear_in2(h_t1)
            h_t2_ = self.word_linear_in2(h_t2)
            h_t1 = h_t1_.view(tgt_batch, 1, tgt_dim)
            h_t2 = h_t2_.view(tgt_batch, 1, tgt_dim)
            return torch.bmm(h_t2, h_s_) - dist_lembda1 * torch.bmm(h_t1, h_s_)
            
        elif type == 'word' and dist_num == "third":
            h_t1_ = self.word_linear_in3(h_t1)
            h_t2_ = self.word_linear_in3(h_t2)
            h_t3_ = self.word_linear_in3(h_t3)
            h_t1 = h_t1_.view(tgt_batch, 1, tgt_dim)
            h_t2 = h_t2_.view(tgt_batch, 1, tgt_dim)
            h_t3 = h_t3_.view(tgt_batch, 1, tgt_dim)
            return torch.bmm(h_t3, h_s_) - dist_lembda1 * torch.bmm(h_t1, h_s_) - dist_lembda2 * torch.bmm(h_t2, h_s_)
            
        elif type == 'sent' and dist_num == "first":
            h_t1_ = self.sent_linear_in1(h_t1)
            h_t1 = h_t1_.view(tgt_batch, 1, tgt_dim)
            return torch.bmm(h_t1, h_s_)

        elif type == 'sent' and dist_num == "second":
            h_t1_ = self.sent_linear_in2(h_t1)
            h_t2_ = self.sent_linear_in2(h_t2)
            h_t1 = h_t1_.view(tgt_batch, 1, tgt_dim)
            h_t2 = h_t2_.view(tgt_batch, 1, tgt_dim)
            return torch.bmm(h_t2, h_s_) - dist_lembda1 * torch.bmm(h_t1, h_s_)
       
        elif type == 'sent' and dist_num == "third":
            h_t1_ = self.sent_linear_in3(h_t1)
            h_t2_ = self.sent_linear_in3(h_t2)
            h_t3_ = self.sent_linear_in3(h_t3)
            h_t1 = h_t1_.view(tgt_batch, 1, tgt_dim)
            h_t2 = h_t2_.view(tgt_batch, 1, tgt_dim)
            h_t3 = h_t3_.view(tgt_batch, 1, tgt_dim)
            return torch.bmm(h_t3, h_s_) - dist_lembda1 * torch.bmm(h_t1, h_s_) - dist_lembda2 * torch.bmm(h_t2, h_s_)
            
        else:
            raise NotImplementedError

    def forward(self, word_bank, word_lengths,
                sent_bank, sent_lengths, static_attn, dist_num, source1, source2=None, source3=None):
        
        word_max_len, word_batch, words_max_len, word_dim = word_bank.size()
        sent_max_len, sent_batch, sent_dim = sent_bank.size()
        assert word_batch == sent_batch
        assert words_max_len == sent_max_len
        target_batch, target_dim = source1.size()

        # reshape for compute word score
        # (word_max_len, word_batch, words_max_len, word_dim) -> transpose
        # (word_batch, word_max_len, words_max_len, word_dim) -> transpose   !!! important, otherwise do not match the src_map
        # (word_batch, words_max_len, word_max_len, word_dim)
        word_bank = word_bank.contiguous().transpose(0, 1).transpose(1, 2).contiguous().view(
            word_batch, words_max_len * word_max_len, word_dim)
        word_align = self.score(word_bank, 'word', dist_num, source1, source2, source3)

        sent_bank = sent_bank.transpose(0, 1).contiguous()
        sent_align = self.score(sent_bank, 'sent', dist_num, source1, source2, source3)        
        
        align = (word_align.view(word_batch, 1, words_max_len, word_max_len) * sent_align.unsqueeze(-1) *\
                     static_attn.unsqueeze(1).unsqueeze(-1)).view(word_batch, 1, words_max_len * word_max_len)
       
        mask = sequence_mask(word_lengths.view(-1), max_len=word_max_len).view(
            word_batch, words_max_len * word_max_len).unsqueeze(1)
        
        align.masked_fill_(~(mask).to(self.device), -float('inf'))
        align_vectors = self.softmax(align) + 1e-20
        c = torch.bmm(align_vectors, word_bank).squeeze(1)
        
        if dist_num == 'first':
            concat_c = torch.cat([c, source1], -1).view(target_batch, target_dim * 2)
        if dist_num == 'second':
            concat_c = torch.cat([c, source2], -1).view(target_batch, target_dim * 2)
        if dist_num == 'third':
            concat_c = torch.cat([c, source3], -1).view(target_batch, target_dim * 2)
                        
        attn_h = self.linear_out(concat_c).view(target_batch, target_dim)
        attn_h = self.tanh(attn_h)

        return attn_h, align_vectors.squeeze(1)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=768, n_heads=12):
        """
        Args:
            embed_dim: dimension of embeding vector output
            n_heads: number of self attention heads
        """
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim  # 512 dim
        self.n_heads = n_heads  # 8
        self.single_head_dim = int(self.embed_dim / self.n_heads)  # 512/8 = 64  . each key,query, value will be of 64d

        # key,query and value matrixes    #64 x 64
        self.query_matrix = nn.Linear(self.single_head_dim, self.single_head_dim,
                                      bias=False)  # single key matrix for all 8 keys #512x512
        self.key_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.value_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.out = nn.Linear(self.n_heads * self.single_head_dim, self.embed_dim)

    def forward(self, query, key, value, mask=None):  # batch_size x seqence_length x embedding_dim
        """
            Args:
                key : key vector
                query : query vector
                value : value vector
                mask: mask for decoder

            Returns:
                output vector from multihead attention
        """

        batch_size = key.size(0)
        seq_length = key.size(1)
        # query dimension can change in decoder during inference.  so we can't take general seq_length
        seq_length_query = query.size(1)

        # batch_size x seq_length x embed_dim

        key = key.view(key.size(1), key.size(0), self.n_heads,self.single_head_dim)  # batch_size x seq_len x n_heads x single_head_dim
        query = query.view(query.size(1), query.size(0), self.n_heads, self.single_head_dim)
        value = value.view(value.size(1), value.size(0), self.n_heads, self.single_head_dim)
        print(f"key shape: {key.shape}")
        print(f"query shape: {query.shape}")
        k = self.key_matrix(key)
        q = self.query_matrix(query)
        v = self.value_matrix(value)
        print(f"v shape: {v.shape}")
        q = q.transpose(1, 2)  # batch_size x n_heads x seq_len x single_head_dim
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        print(f"v shape: {v.shape}")
        print(f"q shape: {q.shape}")

        # compute attentions , # adjusted key for matrix multiplication

        k_adjust = k.transpose(-1, -2)  # batch_size x n_heads x single_head_dim x seqence_length
        print(f"k_adjust shape: {k_adjust.shape}")
        product = torch.matmul(q,k_adjust)  # batch_size x n_heads x seq_len x single_head_dim * #batch_size x n_heads x single_head_dim x seqence_length
        # batch_size x n_heads x seqence_length x seqence_length

        # fill those positions of product matrix as (-1e20) where mask positions are 0

        if mask is not None:
            product = product.masked_fill(mask == 0, -float("inf"))
        # divising by square root of key dimension

        product = product / math.sqrt(self.single_head_dim)  # sqrt(64)

        # applying softmax

        scores = F.softmax(product, dim=-1)
        print(f"score shape: {scores.shape}")
        print(f"v shape: {v.shape}")
        scores = torch.matmul(scores, v)  # batch x n_head x seq_leg x sing_head_dim
        print(f"scores shape: {scores.shape}")
        # concatenated output
        concat = scores.transpose(1, 2).contiguous().view(query.size(1), query.size(0),
                                                          self.single_head_dim * self.n_heads)  # batch x seq_leng x embed_dims

        output = self.out(concat)  # batch x seq_length x embed_dim

        return output

class NegativeAttentionModel(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.multihead_attn_pq = nn.MultiheadAttention(d_model, num_heads)
        self.multihead_attn_pa = nn.MultiheadAttention(d_model, num_heads)

    def forward(self,P,Q,A):
        attn_output_pq , _ = self.multihead_attn_pq(P,Q,Q)
        attn_output_pa , _ = self.multihead_attn_pa(P,A,A)
        attn_output = attn_output_pq - attn_output_pa

        return attn_output