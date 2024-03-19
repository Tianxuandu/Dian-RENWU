import torch
import torch.nn.functional as F


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    计算缩放点积注意力。
    :param Q: 查询 (query)，形状为 (batch_size, num_heads, seq_length, d_k)
    :param K: 键 (key)，形状为 (batch_size, num_heads, seq_length, d_k)
    :param V: 值 (value)，形状为 (batch_size, num_heads, seq_length, d_v)
    :param mask: 可选的掩码，用于屏蔽某些位置的注意力权重
    :return: 多头注意力的输出，形状为 (batch_size, num_heads, seq_length, d_v)
    """
    d_k = torch.tensor(Q.size()[-1], dtype=torch.float32)  # 计算键的维度
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(d_k)  # 计算缩放点积

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))  # 使用掩码屏蔽某些分数

    attention_weights = F.softmax(scores, dim=-1)  # 应用Softmax函数计算注意力权重

    return torch.matmul(attention_weights, V)  # 计算加权和


# 示例：计算一个随机矩阵的注意力权重
batch_size = 2
num_heads = 4
seq_length = 10
d_k = 64
d_v = 64

# 创建随机的查询、键和值矩阵
Q = torch.rand(batch_size, num_heads, seq_length, d_k)
K = torch.rand(batch_size, num_heads, seq_length, d_k)
V = torch.rand(batch_size, num_heads, seq_length, d_v)

# 假设我们不需要掩码
mask = None

# 计算注意力权重
output = scaled_dot_product_attention(Q, K, V, mask=mask)

print(output)  # 输出多头注意力的输出