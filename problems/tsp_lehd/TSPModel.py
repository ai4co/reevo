import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from gpt import heuristics_v2 as heuristics
except:
    from gpt import heuristics


IMPL_REEVO = True

def heuristics_seed(distance_matrix: torch.Tensor) -> torch.Tensor:
    """
    heu_ij = - log(dis_ij) if j is the topK nearest neighbor of i, else - dis_ij
    """
    distance_matrix[distance_matrix == 0] = 1e5
    K = 100
    # Compute top-k nearest neighbors (smallest distances)
    values, indices = torch.topk(distance_matrix, k=K, largest=False, dim=1)
    heu = -distance_matrix.clone()
    # Create a mask where topk indices are True and others are False
    topk_mask = torch.zeros_like(distance_matrix, dtype=torch.bool)
    topk_mask.scatter_(1, indices, True)
    # Apply -log(d_ij) only to the top-k elements
    heu[topk_mask] = -torch.log(distance_matrix[topk_mask])
    return heu

class TSPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.mode = model_params['mode']
        self.encoder = TSP_Encoder(**model_params)
        self.decoder = TSP_Decoder(**model_params)
        self.encoded_nodes = None

    def forward(self, state, selected_node_list, solution, current_step,repair = False):
        # solution's shape : [B, V]
        batch_size_V = state.data.size(0)

        if self.mode == 'train':
            raise NotImplementedError

            probs = self.decoder(self.encoder(state.data), selected_node_list)

            selected_student = probs.argmax(dim=1)  # shape: B
            selected_teacher = solution[:, current_step - 1]  # shape: B
            prob = probs[torch.arange(batch_size_V)[:, None], selected_teacher[:, None]].reshape(batch_size_V, 1)  # shape: [B, 1]

        if self.mode == 'test':
            if  repair == False :
                if current_step <= 1:
                    self.encoded_nodes = self.encoder(state.data) # state.data.shape: [B, V, 2]
                    ######################## ReEvo #############################
                    distance_matrices = torch.cdist(state.data, state.data, p=2)
                    if IMPL_REEVO:
                        self.attention_bias = torch.stack([
                            heuristics(distance_matrices[i]) for i in range(distance_matrices.size(0))
                        ], dim=0)
                        assert not torch.isnan(self.attention_bias).any()
                        assert not torch.isinf(self.attention_bias).any()
                    else:
                        self.attention_bias = None
                    ###########################################################

                # selected_node_list.shape: (batch size, 1) -> (batch size, problem - 1)
                probs = self.decoder(self.encoded_nodes, selected_node_list, attention_bias=self.attention_bias)

                selected_student = probs.argmax(dim=1)
                selected_teacher = selected_student
                prob = 1

            if  repair == True :
                raise NotImplementedError
                if current_step <= 2:
                    self.encoded_nodes = self.encoder(state.data)

                probs = self.decoder(self.encoded_nodes, selected_node_list)

                selected_student = probs.argmax(dim=1)
                selected_teacher = selected_student
                prob = 1

        return selected_teacher, prob, 1, selected_student



########################################
# ENCODER
########################################
class TSP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num =  1
        self.embedding = nn.Linear(2, embedding_dim, bias=True)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])



    def forward(self, data):

        embedded_input = self.embedding(data)
        out = embedded_input
        for layer in self.layers:
            out = layer(out)
        return out


class TSP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['decoder_layer_num']

        self.embedding_first_node = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.embedding_last_node = nn.Linear(embedding_dim, embedding_dim, bias=True)

        self.layers = nn.ModuleList([DecoderLayer(**model_params) for _ in range(encoder_layer_num)])

        self.k_1 = nn.Linear(embedding_dim, embedding_dim, bias=True)

        self.Linear_final = nn.Linear(embedding_dim, 1, bias=True)


    def _get_new_data(self, data, selected_node_list, prob_size, B_V):

        list = selected_node_list

        new_list = torch.arange(prob_size)[None, :].repeat(B_V, 1)

        new_list_len = prob_size - list.shape[1]  # shape: [B, V-current_step]

        index_2 = list.type(torch.long)

        index_1 = torch.arange(B_V, dtype=torch.long)[:, None].expand(B_V, index_2.shape[1])

        new_list[index_1, index_2] = -2

        unselect_list = new_list[torch.gt(new_list, -1)].view(B_V, new_list_len)

        # ----------------------------------------------------------------------------

        new_data = data

        emb_dim = data.shape[-1]

        new_data_len = new_list_len

        index_2_ = unselect_list.repeat_interleave(repeats=emb_dim, dim=1)

        index_1_ = torch.arange(B_V, dtype=torch.long)[:, None].expand(B_V, index_2_.shape[1])

        index_3_ = torch.arange(emb_dim)[None, :].repeat(repeats=(B_V, new_data_len))

        new_data_ = new_data[index_1_, index_2_, index_3_].view(B_V, new_data_len, emb_dim)

        return new_data_, unselect_list

    def _get_encoding(self, encoded_nodes, node_index_to_pick):

        batch_size = node_index_to_pick.size(0)
        pomo_size = node_index_to_pick.size(1)
        embedding_dim = encoded_nodes.size(2)

        gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)

        picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)

        return picked_nodes

    def forward(self, data, selected_node_list, attention_bias=None):
        
        # data.shape = (B, problem, embedding_dim)
        batch_size_V = data.shape[0]  # B

        problem_size = data.shape[1]

        new_data = data

        # selected_node_list.shape: [B, current_step]
        left_encoded_node, unselect_list = self._get_new_data(new_data, selected_node_list, problem_size, batch_size_V)

        first_and_last_node = self._get_encoding(new_data,selected_node_list[:,[0,-1]])
        embedded_first_node_ = first_and_last_node[:,0]
        embedded_last_node_ = first_and_last_node[:,1]

        #------------------------------------------------
        #------------------------------------------------

        embedded_first_node_ = self.embedding_first_node(embedded_first_node_)

        embedded_last_node_ = self.embedding_last_node(embedded_last_node_)

        out = torch.cat((embedded_first_node_.unsqueeze(1), left_encoded_node,embedded_last_node_.unsqueeze(1)), dim=1)

        layer_count=0

        for layer in self.layers:

            out = layer(out)
            layer_count += 1

        out = self.Linear_final(out).squeeze(-1)
        # Linear_final: (B, V, 1) -> (B, V)
        
        # ReEvo: add attention bias
        if IMPL_REEVO:
            # Fetch the last selected node's attention bias for each batch
            current_node_idx = selected_node_list[:, -1]  # shape: (B,)
            attention_bias_current_node = attention_bias[torch.arange(batch_size_V), current_node_idx]  # shape: (B, V)
            attention_bias_current_node_unselect = attention_bias_current_node[torch.arange(batch_size_V)[:, None], unselect_list]  # shape: (B, V-current_step)
            

            out[:, 1:-1] += attention_bias_current_node_unselect
            

        out[:, [0,-1]] = out[:, [0,-1]] + float('-inf')

        props = F.softmax(out, dim=-1)
        props = props[:, 1:-1]

        index_small = torch.le(props, 1e-5)
        props_clone = props.clone()
        props_clone[index_small] = props_clone[index_small] + torch.tensor(1e-7, dtype=props_clone[index_small].dtype)  # prevent the probability from being too small
        props = props_clone

        new_props = torch.zeros(batch_size_V, problem_size)

        index_1_ = torch.arange(batch_size_V, dtype=torch.long)[:, None].expand(batch_size_V, selected_node_list.shape[1])  # shape: [B*(V-1), n]
        index_2_ = selected_node_list.type(torch.long)
        new_props[index_1_, index_2_] = -2
        index = torch.gt(new_props, -1).view(batch_size_V, -1)

        new_props[index] = props.ravel()

        return new_props

class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.feedForward = Feed_Forward_Module(**model_params)


    def forward(self, input1):

        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)

        out_concat = multi_head_attention(q, k, v)

        multi_head_out = self.multi_head_combine(out_concat)

        out1 = input1 + multi_head_out
        out2 = self.feedForward(out1)
        out3 = out1 +  out2
        return out3


class DecoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.feedForward = Feed_Forward_Module(**model_params)


    def forward(self, input1):

        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)

        out_concat = multi_head_attention(q, k, v)

        multi_head_out = self.multi_head_combine(out_concat)

        out1 = input1 + multi_head_out
        out2 = self.feedForward(out1)
        out3 = out1 +  out2
        return out3


def reshape_by_heads(qkv, head_num):

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)

    q_transposed = q_reshaped.transpose(1, 2)

    return q_transposed


def multi_head_attention(q, k, v):

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))  # shape: (B, head_num, n, n)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))

    weights = nn.Softmax(dim=3)(score_scaled)  # shape: (B, head_num, n, n)

    out = torch.matmul(weights, v)  # shape: (B, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)  # shape: (B, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)  # shape: (B, n, head_num*key_dim)

    return out_concat



class Feed_Forward_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))
