import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, d_k, dropout=0.0):
    """
    计算缩放点积注意力分数。
    Q: 查询矩阵 (batch_size, num_heads, seq_length, d_k)
    K: 键矩阵 (batch_size, num_heads, seq_length, d_k)
    V: 值矩阵 (batch_size, num_heads, seq_length, d_v)
    d_k: 键和查询的维度
    dropout: 可选，dropout比率
    """
    batch_size = Q.size(0)
    
    # 计算点积并缩放
    attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    
    # 应用Softmax
    attention_probs = F.softmax(attention_scores, dim=-1)
    
    # 应用dropout
    if dropout > 0:
        attention_probs = F.dropout(attention_probs, p=dropout, dim=-1)
    
    # 计算加权和
    attention_output = torch.matmul(attention_probs, V)
    
    return attention_output

# 示例：计算一个随机矩阵的注意力权重
batch_size = 2
num_heads = 2
seq_length = 4
d_k = 3
d_v = 3

# 创建随机矩阵
Q = torch.rand(batch_size, num_heads, seq_length, d_k)
K = torch.rand(batch_size, num_heads, seq_length, d_k)
V = torch.rand(batch_size, num_heads, seq_length, d_v)

# 计算注意力权重
attention_output = scaled_dot_product_attention(Q, K, V, d_k)

print("注意力权重 (输出):")
print(attention_output)
