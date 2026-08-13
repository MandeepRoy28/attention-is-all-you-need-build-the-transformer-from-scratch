"""
Attention Is All You Need: Build the Transformer From Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_token_to_id_vocab
def build_token_to_id_vocab(sentences, specials=('<pad>', '<bos>', '<eos>', '<unk>')):
    # TODO: build a token-to-id dict with specials first, then corpus tokens in first-seen order.
    token_to_id = {}

    counter = 0
    for i in range(0, len(specials)):
        token_to_id[specials[i]] = counter
        counter+=1

    for s in sentences :
        for w in s.split() :
            if w in token_to_id.keys() :
                continue
            else :
                token_to_id[w] = counter
                counter+=1 

    return token_to_id

# Step 2 - build_id_to_token_vocab
def build_id_to_token_vocab(token_to_id):
    # TODO: build the inverse id-to-token dictionary from token_to_id
    id_to_token = {}

    for k in token_to_id.keys() :
        id_to_token[token_to_id[k]] = k

    return id_to_token

# Step 3 - encode_sentence_to_ids
def encode_sentence_to_ids(sentence, token_to_id, unk_token='<unk>'):
    # TODO: convert whitespace tokens of `sentence` to ids via `token_to_id`, using `unk_token`'s id for OOV
    id_list = []

    for s in sentence.split() :
        if s in token_to_id.keys() :
            id_list.append(token_to_id[s])
        else :
            id_list.append(token_to_id[unk_token])

    return id_list

# Step 4 - decode_ids_to_tokens
def decode_ids_to_tokens(ids, id_to_token):
    # TODO: map each id in ids to its token string via id_to_token and return the list
    id_to_token_list = []

    for _id in ids :
        id_to_token_list.append(id_to_token[_id])

    return id_to_token_list

# Step 5 - pad_id_sequence
def pad_id_sequence(ids, max_len, pad_id):
    # TODO: return a list of length exactly max_len, padding with pad_id or truncating.
    pad_ids = [pad_id]*max_len
    for i in range(0, max_len):
        if(i>=len(ids)):
            continue 
        pad_ids[i] = ids[i]

    return pad_ids

# Step 6 - stack_padded_sequences_to_batch
import torch

def stack_padded_sequences_to_batch(padded_sequences):
    """Stack a list of equal-length padded id sequences into a 2D LongTensor batch."""
    # TODO: stack padded id sequences into a (B, L) torch.long tensor
    padded_sequences_tr = torch.tensor(padded_sequences, dtype=torch.long)
    return padded_sequences_tr

# Step 7 - scale_embeddings_by_sqrt_d_model
import math
import torch

def scale_embeddings_by_sqrt_d_model(embeddings, d_model):
    """Scale a token embedding tensor by sqrt(d_model)."""
    # TODO: rescale embeddings by sqrt(d_model) as in the original Transformer paper
    return embeddings*(math.sqrt(d_model))

# Step 8 - compute_positional_div_term
import torch

def compute_positional_div_term(d_model):
    # TODO: return a 1D FloatTensor of length d_model // 2 holding the sinusoidal frequency divisors
    div_terms = []
    for i in range(0, d_model//2):
        div_terms.append((10000**(-2*i/d_model)))
    return torch.tensor(div_terms, dtype=torch.float32)

# Step 9 - build_position_index_column
import torch

def build_position_index_column(max_len):
    """Return a (max_len, 1) float tensor of [0, 1, ..., max_len-1]."""
    # TODO: build a column vector of position indices from 0 to max_len-1
    rows = torch.arange(max_len, dtype=torch.float32)
    rows = rows.unsqueeze(1)
    return rows

# Step 10 - fill_even_indices_with_sin
import torch

def fill_even_indices_with_sin(pe, position, div_term):
    """Fill even feature indices of pe with sin(position * div_term)."""
    # TODO: write sin(position * div_term) into the even-indexed columns of pe and return it
    sin_term = torch.sin(position*div_term)
    m,n = pe.size()

    pe[:, 0:n:2] = sin_term
    return pe

# Step 11 - fill_odd_indices_with_cos
import torch

def fill_odd_indices_with_cos(pe, position, div_term):
    # TODO: fill the odd-indexed columns of pe with cos(position * div_term)
    cos_term = torch.cos(position*div_term)
    m,n = pe.size()

    pe[:,1:n:2] = cos_term
    return pe

# Step 12 - build_sinusoidal_positional_encoding
import torch

def build_sinusoidal_positional_encoding(max_len, d_model):
    """Assemble the (max_len, d_model) sinusoidal positional encoding matrix."""
    # TODO: build the (max_len, d_model) sinusoidal positional encoding matrix
    pe = torch.zeros(max_len, d_model)

    div_term = compute_positional_div_term(d_model)
    position = build_position_index_column(max_len)

    pe = fill_even_indices_with_sin(pe, position, div_term)
    pe = fill_odd_indices_with_cos(pe, position, div_term)

    return pe

# Step 13 - add_positional_encoding_to_embeddings
import torch

def add_positional_encoding_to_embeddings(embedded_batch, positional_encoding):
    # TODO: add the first L rows of positional_encoding to embedded_batch and return the sum.
    B,L,d_model = embedded_batch.size()
    pos_to_add = positional_encoding[0:L,:]
    embedded_batch = embedded_batch + pos_to_add
    return embedded_batch

# Step 14 - build_padding_mask
import torch

def build_padding_mask(token_ids, pad_id):
    """Return a (B, 1, 1, L) bool mask: True where token_ids != pad_id."""
    # TODO: build a boolean mask marking non-pad positions, shaped for broadcasting against attention scores
    mask = token_ids != pad_id
    mask = mask.unsqueeze(1).unsqueeze(2)
    return mask

# Step 15 - build_causal_mask
import torch

def build_causal_mask(seq_len):
    """Return a (1, 1, seq_len, seq_len) bool mask, True on and below diagonal."""
    # TODO: build a lower-triangular boolean causal mask of shape (1, 1, seq_len, seq_len)
    casual_mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)

    for i in range(0, seq_len):
        for j in range(0, seq_len):
            if j<=i :
                casual_mask[i][j] = True

    return casual_mask.unsqueeze(0).unsqueeze(0)

# Step 16 - combine_padding_and_causal_masks
import torch

def combine_padding_and_causal_masks(padding_mask, causal_mask):
    # TODO: combine a (B,1,1,L) padding mask with a (1,1,L,L) causal mask into (B,1,L,L).
    pad_and_cas_mask = padding_mask & causal_mask
    return pad_and_cas_mask

# Step 17 - compute_raw_attention_scores
import torch

def compute_raw_attention_scores(query, key):
    """Compute raw attention scores Q @ K^T over the last two dimensions."""
    # TODO: matmul query with the transpose of key over the last two axes
    return query@key.transpose(-2,-1)

# Step 18 - scale_attention_scores
import torch
import math

def scale_attention_scores(scores, d_k):
    # TODO: divide raw attention scores by sqrt(d_k) to stabilize softmax inputs
    return scores/(math.sqrt(d_k))

# Step 19 - mask_attention_scores_with_neg_inf
import torch

def mask_attention_scores_with_neg_inf(scores, mask):
    """Set entries of scores where mask is False to -inf."""
    # TODO: replace blocked positions of scores with negative infinity
    masked_score = scores.masked_fill(~mask, float('-inf'))
    return masked_score

# Step 20 - softmax_attention_weights
import torch

def softmax_attention_weights(masked_scores):
    # TODO: softmax over the last axis, zeroing rows that are entirely -inf
    P = torch.softmax(masked_scores, dim=-1)
    P = torch.nan_to_num(P, nan=0.0)
    return P

# Step 21 - apply_attention_weights_to_values
import torch

def apply_attention_weights_to_values(attention_weights, value):
    """Multiply attention weights by the value matrix to produce context vectors."""
    # TODO: combine attention weights (..., Lq, Lk) with value (..., Lk, d_v)
    return attention_weights @ value

# Step 22 - scaled_dot_product_attention
import torch

def scaled_dot_product_attention(query, key, value, mask=None):
    """Run scaled dot-product attention; return (context, attention_weights)."""
    # TODO: chain raw scores, scale by sqrt(d_k), optionally mask, softmax, then mix values
    d_k = query.size(-1)
    S = compute_raw_attention_scores(query, key)
    S = scale_attention_scores(S, d_k)

    if mask is not None :
        S = mask_attention_scores_with_neg_inf(S, mask)

    attention_weight = softmax_attention_weights(S)

    context = apply_attention_weights_to_values(attention_weight, value)

    return (context, attention_weight)

# Step 23 - split_last_dim_into_heads
import torch

def split_last_dim_into_heads(tensor, num_heads):
    # TODO: reshape (B, L, d_model) into (B, L, num_heads, d_model // num_heads)
    B, L, d_model = tensor.size()
    return tensor.view(B, L, num_heads, d_model//num_heads)

# Step 24 - transpose_heads_before_sequence
import torch

def transpose_heads_before_sequence(split_tensor):
    # TODO: rearrange (B, L, num_heads, d_k) into (B, num_heads, L, d_k).
    return split_tensor.transpose(1,2)

# Step 25 - merge_heads_back_to_model_dim
import torch

def merge_heads_back_to_model_dim(multi_head_tensor):
    # TODO: merge the head axis back into the feature axis to reconstruct d_model
    multi_head_tensor_heads = multi_head_tensor.transpose(1,2)
    return multi_head_tensor_heads.reshape(*multi_head_tensor_heads.shape[:-2], -1)

# Step 26 - apply_linear_projection
def apply_linear_projection(x, weight, bias):
    # TODO: return x @ weight^T + bias (bias may be None) with shape (..., out_features)
    y = x @ weight.T 

    if bias is not None :
        y = y + bias
    
    return y

# Step 27 - project_to_query_key_value
def project_to_query_key_value(x, w_q, b_q, w_k, b_k, w_v, b_v):
    # TODO: project x into separate query, key, and value tensors via three linear layers
    Q = x@w_q.T  
    if b_q is not None :
        Q = Q + b_q

    K = x@w_k.T
    if b_k is not None :
        K = K + b_k

    V = x@w_v.T
    if b_v is not None :
        V = V + b_v 

    return Q, K, V

# Step 28 - split_qkv_into_heads
import torch

def split_qkv_into_heads(q, k, v, num_heads):
    # TODO: split each of q, k, v into (B, num_heads, L, d_k) and return as a tuple
    q_h = transpose_heads_before_sequence(split_last_dim_into_heads(q, num_heads))
    k_h = transpose_heads_before_sequence(split_last_dim_into_heads(k, num_heads))
    v_h = transpose_heads_before_sequence(split_last_dim_into_heads(v, num_heads))

    return (q_h, k_h, v_h)

# Step 29 - multi_head_scaled_dot_product_attention
import torch

def multi_head_scaled_dot_product_attention(q_h, k_h, v_h, mask=None):
    # TODO: run scaled dot-product attention over per-head Q, K, V and return (context, weights)
    return scaled_dot_product_attention(q_h, k_h, v_h, mask)

# Step 30 - merge_heads_and_project_output
import torch

def merge_heads_and_project_output(context, w_o, b_o):
    # TODO: merge the head axis back into d_model and apply the output linear projection.
    X = merge_heads_back_to_model_dim(context)
    return apply_linear_projection(X, w_o, b_o)

# Step 31 - assemble_multi_head_attention_forward
def assemble_multi_head_attention_forward(query, key, value, w_q, w_k, w_v, w_o, num_heads, mask=None):
    # TODO: project Q/K/V, split into heads, run scaled dot-product attention, merge heads, output projection.
    Q = apply_linear_projection(query, w_q, None)
    K = apply_linear_projection(key, w_k, None)
    V = apply_linear_projection(value, w_v, None)

    Q, K, V = split_qkv_into_heads(Q, K, V, num_heads)

    context, attention_weight = multi_head_scaled_dot_product_attention(Q, K, V, mask)
    
    O = merge_heads_and_project_output(context, w_o, None)
    
    return O

# Step 32 - apply_ffn_first_linear_and_relu
import torch
def apply_ffn_first_linear_and_relu(x, w1, b1):
    # TODO: project x by w1, add b1, then apply a ReLU activation.
    y = x @ w1 + b1 
    y = torch.where(y<0, 0.0, y)
    return y

# Step 33 - apply_ffn_second_linear
import torch

def apply_ffn_second_linear(hidden, w2, b2):
    # TODO: project hidden (..., d_ff) back to (..., d_model) via w2 and b2.
    return hidden@w2 + b2

# Step 34 - position_wise_feed_forward_network
def position_wise_feed_forward_network(x, w1, b1, w2, b2):
    # TODO: compose the two FFN linears with a ReLU in between, returning shape (B, T, d_model).
    layer1 = apply_ffn_first_linear_and_relu(x, w1, b1)
    layer2 = apply_ffn_second_linear(layer1, w2, b2)
    return layer2

# Step 35 - compute_layer_norm_mean_and_variance
import torch

def compute_layer_norm_mean_and_variance(x):
    # TODO: return (mean, variance) reduced over the last dim with shape (..., 1)
    mean = torch.mean(x, dim=-1, keepdim=True)
    var = torch.var(x, correction=0, dim=-1, keepdim=True)

    return mean, var

# Step 36 - normalize_and_scale_with_gamma_beta
import torch

def normalize_and_scale_with_gamma_beta(x, gamma, beta, eps=1e-5):
    # TODO: standardize x along the last axis then apply gamma and beta affine transform
    mean, var = compute_layer_norm_mean_and_variance(x)
    x_cap = (x-mean)/(torch.sqrt(var + torch.tensor(eps, dtype=torch.float32)))

    y = x_cap * gamma + beta
    return y

# Step 37 - apply_residual_add_and_norm
import torch

def apply_residual_add_and_norm(residual_input, sublayer_output, gamma, beta, eps=1e-5):
    # TODO: combine the residual with the sublayer output and layer-normalize the result.
    return normalize_and_scale_with_gamma_beta(residual_input+sublayer_output, gamma, beta, eps)

# Step 38 - apply_dropout_with_keep_mask
def apply_dropout_with_keep_mask(x, keep_mask, keep_prob):
    # TODO: multiply x by the boolean keep_mask and rescale by 1/keep_prob.
    return (x*keep_mask)/keep_prob

# Step 39 - encoder_layer_self_attention_sublayer
def encoder_layer_self_attention_sublayer(x, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    # TODO: run multi-head self-attention on x and wrap with residual add-and-norm.
    O = assemble_multi_head_attention_forward(x, x, x, w_q, w_k, w_v, w_o, num_heads, src_mask)
    
    return apply_residual_add_and_norm(x,O, gamma, beta, eps=1e-5)

# Step 40 - encoder_layer_feed_forward_sublayer
def encoder_layer_feed_forward_sublayer(x, w1, b1, w2, b2, gamma, beta):
    # TODO: run the position-wise FFN on x and wrap it with residual add-and-norm.
    sublayer_output = position_wise_feed_forward_network(x, w1, b1, w2, b2)
    return apply_residual_add_and_norm(x, sublayer_output, gamma, beta)

# Step 41 - assemble_encoder_layer
def assemble_encoder_layer(x, layer_params, num_heads, src_mask):
    # TODO: chain the self-attention sublayer and the feed-forward sublayer using layer_params.
    block1 = encoder_layer_self_attention_sublayer(
        x, layer_params['w_q'], layer_params['w_k'], layer_params['w_v'], layer_params['w_o'], layer_params['attn_gamma'], layer_params['attn_beta'], num_heads, src_mask
        )

    block2 = encoder_layer_feed_forward_sublayer(
        block1, layer_params['w1'], layer_params['b1'], layer_params['w2'], layer_params['b2'], layer_params['ffn_gamma'], layer_params['ffn_beta']
        )

    return block2

# Step 42 - stack_encoder_layers
def stack_encoder_layers(x, encoder_layer_params_list, num_heads, src_mask):
    # TODO: sequentially apply each encoder layer to the running hidden state and return the final tensor.
    
    hidden = x 

    for layer_params in encoder_layer_params_list :
        hidden = assemble_encoder_layer(hidden, layer_params, num_heads, src_mask)

    return hidden

# Step 43 - decoder_layer_masked_self_attention_sublayer
import torch

def decoder_layer_masked_self_attention_sublayer(y, w_q, w_k, w_v, w_o, gamma, beta, num_heads, tgt_mask):
    # TODO: run masked multi-head self-attention on y and wrap with residual add-and-norm.
    y_attn = assemble_multi_head_attention_forward(y, y, y, w_q, w_k, w_v, w_o, num_heads, tgt_mask)
    y_res_nrm = apply_residual_add_and_norm(y, y_attn, gamma, beta)
    return y_res_nrm

# Step 44 - decoder_layer_cross_attention_sublayer
import torch

def decoder_layer_cross_attention_sublayer(y, encoder_output, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    # TODO: run multi-head cross-attention (Q from y, K/V from encoder_output) and wrap with add-and-norm
    if src_mask is not None and src_mask.dim() == 2:
        src_mask = src_mask.unsqueeze(1).unsqueeze(2)
    head = assemble_multi_head_attention_forward(y, encoder_output, encoder_output, w_q, w_k, w_v, w_o, num_heads, src_mask)
    head_res_norm = apply_residual_add_and_norm(y, head, gamma, beta)
    return head_res_norm

# Step 45 - decoder_layer_feed_forward_sublayer
import torch

def decoder_layer_feed_forward_sublayer(y, w1, b1, w2, b2, gamma, beta):
    # TODO: run the position-wise FFN on y and wrap it with residual add-and-norm
    y_frwd = position_wise_feed_forward_network(y, w1, b1, w2, b2)
    y_res = apply_residual_add_and_norm(y, y_frwd, gamma, beta)
    return y_res

# Step 46 - assemble_decoder_layer
def assemble_decoder_layer(y, encoder_output, layer_params, num_heads, src_mask, tgt_mask):
    """Run a full decoder layer: masked self-attention, cross-attention, then FFN."""
    # TODO: chain the three decoder sublayers using params from layer_params.
    y_msk_attn = decoder_layer_masked_self_attention_sublayer(
        y, layer_params["w_q_self"], layer_params["w_k_self"], layer_params["w_v_self"], layer_params["w_o_self"], layer_params["self_gamma"], layer_params["self_beta"], num_heads, tgt_mask
        )

    xy_attn = decoder_layer_cross_attention_sublayer(
        y_msk_attn, encoder_output, layer_params["w_q_cross"], layer_params["w_k_cross"], layer_params["w_v_cross"], layer_params["w_o_cross"], layer_params["cross_gamma"], layer_params["cross_beta"], num_heads, src_mask
        )

    out = decoder_layer_feed_forward_sublayer(
        xy_attn, layer_params["w1"], layer_params["b1"], layer_params["w2"], layer_params["b2"], layer_params["ffn_gamma"], layer_params["ffn_beta"]
        )

    return out

# Step 47 - stack_decoder_layers
def stack_decoder_layers(y, encoder_output, decoder_layer_params_list, num_heads, src_mask, tgt_mask):
    # TODO: sequentially apply each decoder layer to the running target hidden state.
    hidden = y

    for layer_params in decoder_layer_params_list :
        hidden = assemble_decoder_layer(hidden, encoder_output, layer_params, num_heads, src_mask, tgt_mask)

    return hidden

# Step 48 - apply_final_output_projection
def apply_final_output_projection(decoder_output, output_projection_weight, output_projection_bias=None):
    # TODO: project decoder hidden states (B, T, D) to vocabulary logits (B, T, V).
    output_projection_weight_tp = output_projection_weight.transpose(-1, -2)

    logits = decoder_output@output_projection_weight_tp
    if output_projection_bias is not None :
        logits = logits + output_projection_bias
    
    return logits

# Step 49 - tie_output_projection_to_token_embeddings
import torch

def tie_output_projection_to_token_embeddings(token_embedding_weight):
    """Return an output projection weight that shares storage with token_embedding_weight.

    Input shape: (vocab_size, d_model). Output shape: (d_model, vocab_size).
    """
    # TODO: return an output projection weight tied to the token embedding matrix
    return token_embedding_weight.permute(-1,-2)

# Step 50 - apply_log_softmax_over_vocab
def apply_log_softmax_over_vocab(logits):
    # TODO: Convert decoder logits (B, T, V) into log probabilities over the vocabulary axis.
    return torch.log_softmax(logits, dim=-1)

# Step 51 - run_transformer_forward
def run_transformer_forward(src_ids, tgt_ids, model_params, num_heads, pad_id):
    # TODO: embed src+tgt, add PE, build masks, run encoder/decoder, project to log probs.
    token_embedding = model_params['token_embedding']
    encoder_layers = model_params['encoder_layers']
    decoder_layers = model_params['decoder_layers']
    output_projection = model_params['output_projection']

    # Source and Target Embedding
    src_embd = token_embedding[src_ids]
    tgt_embd = token_embedding[tgt_ids]

    # Scale Emdedding 
    d_model = token_embedding.size(1)
    src_embd = scale_embeddings_by_sqrt_d_model(src_embd, d_model)
    tgt_embd = scale_embeddings_by_sqrt_d_model(tgt_embd, d_model)

    # Positional Encoding 
    src_pe = build_sinusoidal_positional_encoding(src_ids.size(1), d_model)
    tgt_pe = build_sinusoidal_positional_encoding(tgt_ids.size(1), d_model)

    src_embd = add_positional_encoding_to_embeddings(src_embd, src_pe)
    tgt_embd = add_positional_encoding_to_embeddings(tgt_embd, tgt_pe)

    # Masking 
    src_mask = build_padding_mask(src_ids, pad_id)

    tgt_padding_mask = build_padding_mask(tgt_ids, pad_id)
    tgt_casual_mask = build_causal_mask(tgt_ids.size(1))
    tgt_mask = combine_padding_and_causal_masks(tgt_padding_mask, tgt_casual_mask)


    # Encoder 
    encoder_output = stack_encoder_layers(src_embd, encoder_layers, num_heads, src_mask)

    # Decoder 
    decoder_output = stack_decoder_layers(tgt_embd, encoder_output, decoder_layers, num_heads, src_mask, tgt_mask)

    # Logit 
    logit = apply_final_output_projection(decoder_output, output_projection)

    # logsoft_max 
    out = apply_log_softmax_over_vocab(logit)

    return out

# Step 52 - init_encoder_layer_parameters
import torch
import math

def init_encoder_layer_parameters(d_model, num_heads, d_ff):
    """Return a dict of leaf tensors with requires_grad=True for one encoder layer."""
    # TODO: allocate w_q, w_k, w_v, w_o, w1, b1, w2, b2, attn_gamma, attn_beta, ffn_gamma, ffn_beta.
    w_q = torch.normal(mean=0.0, std=0.02, size=(d_model, d_model), dtype=torch.float32, requires_grad=True)
    w_k = torch.normal(mean=0.0, std=0.02, size=(d_model, d_model), dtype=torch.float32, requires_grad=True)
    w_v = torch.normal(mean=0.0, std=0.02, size=(d_model, d_model), dtype=torch.float32, requires_grad=True)
    w_o = torch.normal(mean=0.0, std=0.02, size=(d_model, d_model), dtype=torch.float32, requires_grad=True)

    attn_gamma = torch.ones(d_model, dtype=torch.float32, requires_grad=True)
    attn_beta = torch.zeros(d_model, dtype=torch.float32, requires_grad=True)

    w1 = torch.normal(mean=0.0, std=0.02, size=(d_model, d_ff), dtype=torch.float32, requires_grad=True)
    b1 = torch.zeros(d_ff, dtype=torch.float32, requires_grad=True)
    w2 = torch.normal(mean=0.0, std=0.02, size=(d_ff, d_model), dtype=torch.float32, requires_grad=True)
    b2 = torch.zeros(d_model, dtype=torch.float32, requires_grad=True)

    ffn_gamma = torch.ones(d_model, dtype=torch.float32, requires_grad=True)
    ffn_beta = torch.zeros(d_model, dtype=torch.float32, requires_grad=True)

    return {
        'w_q' : w_q,
        'w_k' : w_k,
        'w_v' : w_v,
        'w_o' : w_o,
        'w1' : w1,
        'b1' : b1,
        'w2' : w2,
        'b2' : b2,
        'attn_gamma' : attn_gamma,
        'attn_beta' : attn_beta,
        'ffn_gamma' : ffn_gamma,
        'ffn_beta' : ffn_beta
    }

# Step 53 - init_decoder_layer_parameters
import torch

def init_decoder_layer_parameters(d_model, num_heads, d_ff):
    # TODO: return a dict of requires_grad tensors for one decoder layer
    w_q_self = torch.normal(mean=0.0, std=0.02, size=(d_model, d_model), dtype=torch.float32, requires_grad=True)
    w_k_self = torch.normal(mean=0.0, std=0.02, size=(d_model, d_model), dtype=torch.float32, requires_grad=True)
    w_v_self = torch.normal(mean=0.0, std=0.02, size=(d_model, d_model), dtype=torch.float32, requires_grad=True)
    w_o_self = torch.normal(mean=0.0, std=0.02, size=(d_model, d_model), dtype=torch.float32, requires_grad=True)

    self_gamma = torch.ones(d_model, dtype=torch.float32, requires_grad=True)
    self_beta = torch.zeros(d_model, dtype=torch.float32, requires_grad=True)

    w_q_cross = torch.normal(mean=0.0, std=0.02, size=(d_model, d_model), dtype=torch.float32, requires_grad=True)
    w_k_cross = torch.normal(mean=0.0, std=0.02, size=(d_model, d_model), dtype=torch.float32, requires_grad=True)
    w_v_cross = torch.normal(mean=0.0, std=0.02, size=(d_model, d_model), dtype=torch.float32, requires_grad=True)
    w_o_cross = torch.normal(mean=0.0, std=0.02, size=(d_model, d_model), dtype=torch.float32, requires_grad=True)

    cross_gamma = torch.ones(d_model, dtype=torch.float32, requires_grad=True)
    cross_beta = torch.zeros(d_model, dtype=torch.float32, requires_grad=True)

    w1 = torch.normal(mean=0.0, std=0.02, size=(d_model, d_ff), dtype=torch.float32, requires_grad=True)
    b1 = torch.zeros(d_ff, dtype=torch.float32, requires_grad=True)
    w2 = torch.normal(mean=0.0, std=0.02, size=(d_ff, d_model), dtype=torch.float32, requires_grad=True)
    b2 = torch.zeros(d_model, dtype=torch.float32, requires_grad=True)

    ffn_gamma = torch.ones(d_model, dtype=torch.float32, requires_grad=True)
    ffn_beta = torch.zeros(d_model, dtype=torch.float32, requires_grad=True)

    return {
        'w_q_self' : w_q_self,
        'w_k_self' : w_k_self,
        'w_v_self' : w_v_self,
        'w_o_self' : w_o_self,
        'self_gamma' : self_gamma,
        'self_beta' : self_beta,
        'w_q_cross' : w_q_cross,
        'w_k_cross' : w_k_cross,
        'w_v_cross' : w_v_cross,
        'w_o_cross' : w_o_cross,
        'cross_gamma' : cross_gamma,
        'cross_beta' : cross_beta,
        'w1' : w1,
        'b1' : b1,
        'w2' : w2,
        'b2' : b2,
        'ffn_gamma' : ffn_gamma,
        'ffn_beta' : ffn_beta
    }

# Step 54 - init_embedding_and_projection_parameters
import torch

def init_embedding_and_projection_parameters(vocab_size, d_model, tie_weights=True):
    """Allocate src/tgt embeddings and output projection (optionally tied)."""
    # TODO: allocate three (vocab_size, d_model) tensors with requires_grad=True
    src_embedding = torch.randn(size=(vocab_size, d_model), dtype=torch.float32, requires_grad=True)
    tgt_embedding = torch.randn(size=(vocab_size, d_model), dtype=torch.float32, requires_grad=True)

    if tie_weights :
        output_projection = tgt_embedding
    else :
        output_projection = torch.randn(size=(vocab_size, d_model), dtype=torch.float32, requires_grad=True)

    return {
        'src_embedding' : src_embedding,
        'tgt_embedding' : tgt_embedding,
        'output_projection' : output_projection
    }

# Step 55 - collect_model_parameters_into_list
import torch

def collect_model_parameters_into_list(encoder_layer_params, decoder_layer_params, embedding_params):
    # TODO: walk the encoder, decoder, and embedding dicts and return a flat deduped list of tensors
    seen = set()
    param_list = []

    def if_seen(p) :
        if id(p) in seen :
            return 
        else :
            param_list.append(p)
            seen.add(id(p))

    for layer in encoder_layer_params :
        for p in layer.values():
            if_seen(p)

    for layer in decoder_layer_params :
        for p in layer.values():
            if_seen(p)

    for p in embedding_params.values():
        if_seen(p)

    return param_list

# Step 56 - shift_targets_right_with_start_token
def shift_targets_right_with_start_token(target_ids, start_token_id):
    # TODO: prepend start_token_id and drop the last column so output shape matches target_ids
    shift = target_ids.clone()
    shift[:,0] = start_token_id
    shift[:,1:] = target_ids[:,0:-1]
    return shift

# Step 57 - compute_noam_learning_rate
import math
def compute_noam_learning_rate(step, d_model, warmup_steps):
    # TODO: return the Noam warmup learning rate for the given step.
    return (1/math.sqrt(d_model))*(min((1/math.sqrt(step)), (step*(1/(warmup_steps**1.5)))))

# Step 58 - build_uniform_smoothing_distribution
import torch

def build_uniform_smoothing_distribution(shape, vocab_size, epsilon):
    # TODO: return a float tensor of `shape` filled with epsilon / (vocab_size - 2).
    distributor = torch.full(size=shape, fill_value=(epsilon/(vocab_size-2)))
    return distributor

# Step 59 - set_confidence_on_gold_tokens
import torch

def set_confidence_on_gold_tokens(smoothed_distribution, gold_token_ids, confidence):
    """Place confidence mass at gold-token positions of a smoothed target distribution."""
    # TODO: write the confidence value at each gold token id along the vocab axis
    golden_distribution = smoothed_distribution.clone()
    batch = torch.arange(smoothed_distribution.size(0)).unsqueeze(1)
    tgt_seq = torch.arange(smoothed_distribution.size(1)).unsqueeze(0)
    golden_distribution[batch,tgt_seq,gold_token_ids] = confidence

    return golden_distribution

# Step 60 - zero_pad_column_and_pad_token_rows (not yet solved)
# TODO: implement

# Step 61 - compute_label_smoothed_kl_loss (not yet solved)
# TODO: implement

# Step 62 - average_loss_over_non_pad_tokens (not yet solved)
# TODO: implement

# Step 63 - compute_token_accuracy_ignoring_pad (not yet solved)
# TODO: implement

# Step 64 - initialize_adam_optimizer_state (not yet solved)
# TODO: implement

# Step 65 - update_adam_first_moment (not yet solved)
# TODO: implement

# Step 66 - update_adam_second_moment (not yet solved)
# TODO: implement

# Step 67 - apply_adam_bias_correction (not yet solved)
# TODO: implement

# Step 69 - apply_adam_step_to_all_parameters (not yet solved)
# TODO: implement

# Step 70 - zero_all_parameter_gradients (not yet solved)
# TODO: implement

# Step 71 - compute_batch_training_loss (not yet solved)
# TODO: implement

# Step 72 - run_training_step_with_backprop (not yet solved)
# TODO: implement

# Step 73 - run_training_loop_for_steps (not yet solved)
# TODO: implement

# Step 74 - pick_next_token_by_argmax (not yet solved)
# TODO: implement

# Step 75 - compute_length_penalty (not yet solved)
# TODO: implement

# Step 76 - compute_candidate_scores (not yet solved)
# TODO: implement

# Step 77 - select_top_k_candidates (not yet solved)
# TODO: implement

# Step 78 - append_tokens_to_beam_sequences (not yet solved)
# TODO: implement

# Step 79 - mark_finished_beams (not yet solved)
# TODO: implement

# Step 80 - select_best_finished_beam (not yet solved)
# TODO: implement

