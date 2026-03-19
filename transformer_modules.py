
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 1. Multi-Head Self-Attention (多头自注意力机制)
class MultiHeadAttention(nn.Module):
    """
    实现多头自注意力机制。
    输入：查询 (Q), 键 (K), 值 (V)。
    输出：经过多头注意力计算后的上下文向量。
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_k = d_model // num_heads  # 每个头的维度
        self.num_heads = num_heads       # 头数
        self.d_model = d_model           # 模型总维度

        # 用于生成 Q, K, V 的线性变换层
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        # 最终输出的线性变换层
        self.output_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # query, key, value 的形状: (batch_size, seq_len, d_model)
        batch_size = query.size(0)

        # 1) 对 Q, K, V 进行线性变换并分成多头
        # 结果形状: (batch_size, num_heads, seq_len, d_k)
        query, key, value = [
            l(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]

        # 2) 计算缩放点积注意力
        # scores 形状: (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 应用掩码 (如果提供)。掩码通常用于防止注意力关注到填充部分或未来信息。
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9) # 将掩码为0的位置填充一个非常小的负数

        # 对分数进行 softmax 归一化，得到注意力权重
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)

        # 3) 将注意力权重应用于值 (V)
        # x 形状: (batch_size, num_heads, seq_len, d_k)
        x = torch.matmul(p_attn, value)

        # 4) 拼接多头并将结果通过最终线性层
        # 结果形状: (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.output_linear(x)

# 2. Feed-Forward Network (前馈网络)
class PositionwiseFeedForward(nn.Module):
    """
    实现位置感知前馈网络 (FFN)。
    包含两个线性变换和一个 ReLU 激活函数。
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)  # 第一个线性层，将 d_model 映射到 d_ff
        self.w_2 = nn.Linear(d_ff, d_model)  # 第二个线性层，将 d_ff 映射回 d_model
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x 的形状: (batch_size, seq_len, d_model)
        # 经过第一个线性层和 ReLU 激活函数
        # 结果形状: (batch_size, seq_len, d_ff)
        # 经过 dropout
        # 结果形状: (batch_size, seq_len, d_model)
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

# 3. Add & Norm (残差连接与层归一化)
class AddAndNorm(nn.Module):
    """
    实现残差连接 (Residual Connection) 和层归一化 (Layer Normalization)。
    """
    def __init__(self, size, dropout=0.1):
        super(AddAndNorm, self).__init__()
        self.norm = nn.LayerNorm(size) # 层归一化
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # x 是子层的输入，sublayer 是要应用的函数 (例如 MultiHeadAttention 或 FFN)
        # 先对子层的输出进行 dropout，然后与原始输入 x 相加 (残差连接)
        # 最后进行层归一化
        return self.norm(x + self.dropout(sublayer(x)))

# 4. Positional Encoding (位置编码)
class PositionalEncoding(nn.Module):
    """
    实现位置编码，将位置信息添加到词嵌入中。
    使用正弦和余弦函数。
    """
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 计算位置编码矩阵
        pe = torch.zeros(max_len, d_model) # 形状: (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1) # 形状: (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)) # 形状: (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term) # 偶数维度使用 sin
        pe[:, 1::2] = torch.cos(position * div_term) # 奇数维度使用 cos
        pe = pe.unsqueeze(0) # 增加一个 batch 维度，形状: (1, max_len, d_model)
        self.register_buffer('pe', pe) # 将 pe 注册为 buffer，它不是模型参数，但需要保存状态

    def forward(self, x):
        # x 的形状: (batch_size, seq_len, d_model)
        # 将位置编码添加到输入 x 中
        # 注意：pe 是 (1, max_len, d_model)，x 是 (batch_size, seq_len, d_model)
        # pe[: , :x.size(1)] 会自动广播到 batch_size
        x = x + self.pe[:, :x.size(1)].requires_grad_(False) # 位置编码不参与梯度计算
        return self.dropout(x)

# 5. Encoder Layer (编码器层)
class EncoderLayer(nn.Module):
    """
    实现一个 Transformer 编码器层。
    包含一个多头自注意力子层和一个前馈网络子层，每个子层后都跟着残差连接和层归一化。
    """
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout) # 多头自注意力
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout) # 前馈网络
        # 两个 AddAndNorm 子层，分别用于自注意力和前馈网络
        self.sublayer = nn.ModuleList([AddAndNorm(d_model, dropout) for _ in range(2)])
        self.d_model = d_model

    def forward(self, x, mask):
        # x 的形状: (batch_size, seq_len, d_model)
        # mask 的形状: (batch_size, 1, 1, seq_len) 或 (batch_size, 1, seq_len, seq_len)

        # 第一个子层：自注意力 -> 残差连接 -> 层归一化
        # sublayer[0] 接收 x 和一个 lambda 函数，该函数封装了 self_attn 的 forward 调用
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))

        # 第二个子层：前馈网络 -> 残差连接 -> 层归一化
        # sublayer[1] 接收 x 和一个 lambda 函数，该函数封装了 feed_forward 的 forward 调用
        return self.sublayer[1](x, self.feed_forward)

# 6. Decoder Layer (解码器层) - 可选但建议实现
class DecoderLayer(nn.Module):
    """
    实现一个 Transformer 解码器层。
    包含一个 Masked Multi-Head Self-Attention 子层、一个 Encoder-Decoder Attention 子层
    和一个前馈网络子层，每个子层后都跟着残差连接和层归一化。
    """
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout) # Masked 自注意力
        self.src_attn = MultiHeadAttention(d_model, num_heads, dropout)  # Encoder-Decoder 注意力
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout) # 前馈网络
        # 三个 AddAndNorm 子层
        self.sublayer = nn.ModuleList([AddAndNorm(d_model, dropout) for _ in range(3)])
        self.d_model = d_model

    def forward(self, x, memory, src_mask, tgt_mask):
        # x 是目标序列的输入 (解码器当前层的输出)
        # memory 是编码器最后一层的输出
        # src_mask 用于编码器输出的注意力 (通常是 padding mask)
        # tgt_mask 用于解码器自注意力 (通常是 look-ahead mask 和 padding mask)

        m = memory # 方便引用

        # 第一个子层：Masked 自注意力 -> 残差连接 -> 层归一化
        # query, key, value 都来自解码器自身的输入 x
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))

        # 第二个子层：Encoder-Decoder 注意力 -> 残差连接 -> 层归一化
        # query 来自解码器 (x)，key 和 value 来自编码器输出 (m)
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))

        # 第三个子层：前馈网络 -> 残差连接 -> 层归一化
        return self.sublayer[2](x, self.feed_forward)


# 辅助函数：生成注意力掩码
def subsequent_mask(size):
    """
    生成一个上三角矩阵，用于在解码器中掩盖未来位置的信息。
    形状: (1, size, size)
    """
    attn_shape = (1, size, size)
    # torch.triu 返回矩阵的上三角部分，k=1 表示从对角线右侧一个位置开始
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8) == 0
    return subsequent_mask


# 简单的词嵌入层
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model) # 查找表，将词汇索引映射到 d_model 维向量
        self.d_model = d_model

    def forward(self, x):
        # x 的形状: (batch_size, seq_len)
        # 结果形状: (batch_size, seq_len, d_model)
        return self.lut(x) * math.sqrt(self.d_model) # 乘以 sqrt(d_model) 进行缩放


# 最终的线性输出层 (用于预测词汇)
class Generator(nn.Module):
    """
    标准线性层 + softmax 生成器。
    """
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        # x 的形状: (batch_size, seq_len, d_model)
        # 结果形状: (batch_size, seq_len, vocab_size)
        return F.log_softmax(self.proj(x), dim=-1)


# 完整的 Transformer 模型 (简化版)
class Transformer(nn.Module):
    """
    一个简化的 Transformer 模型，包含编码器和解码器。
    """\n    def __init__(self, src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, num_heads=8, dropout=0.1):
        super(Transformer, self).__init__()
        # 编码器部分
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(N)])
        self.encoder_norm = nn.LayerNorm(d_model)

        # 解码器部分
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(N)])
        self.decoder_norm = nn.LayerNorm(d_model)

        # 词嵌入和位置编码
        self.src_embed = nn.Sequential(Embeddings(d_model, src_vocab), PositionalEncoding(d_model, dropout))
        self.tgt_embed = nn.Sequential(Embeddings(d_model, tgt_vocab), PositionalEncoding(d_model, dropout))

        # 最终的输出生成器
        self.generator = Generator(d_model, tgt_vocab)

        # 初始化参数
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask):
        # src 的形状: (batch_size, src_seq_len)
        # src_mask 的形状: (batch_size, 1, 1, src_seq_len)
        x = self.src_embed(src) # 词嵌入 + 位置编码
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return self.encoder_norm(x)

    def decode(self, tgt, memory, src_mask, tgt_mask):
        # tgt 的形状: (batch_size, tgt_seq_len)
        # memory 是编码器输出
        # src_mask, tgt_mask 同上
        x = self.tgt_embed(tgt) # 词嵌入 + 位置编码
        for layer in self.decoder_layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.decoder_norm(x)

    def forward(self, src, tgt, src_mask, tgt_mask):
        # 完整的 Transformer 前向传播
        # 首先编码源序列
        memory = self.encode(src, src_mask)
        # 然后解码目标序列
        output = self.decode(tgt, memory, src_mask, tgt_mask)
        # 最后通过生成器得到最终输出
        return self.generator(output)


# 示例用法 (简单的测试)
if __name__ == '__main__':
    # 定义模型参数
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    d_model = 512
    num_heads = 8
    d_ff = 2048
    num_layers = 6
    dropout = 0.1

    # 实例化 Transformer 模型
    model = Transformer(src_vocab_size, tgt_vocab_size, N=num_layers, d_model=d_model, d_ff=d_ff, num_heads=num_heads, dropout=dropout)

    # 创建随机输入数据
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 8

    src_data = torch.randint(0, src_vocab_size, (batch_size, src_seq_len)) # 源序列
    tgt_data = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len)) # 目标序列

    # 创建掩码
    # 源序列掩码 (假设没有 padding，所以全为 True)
    src_mask = (src_data != 0).unsqueeze(1).unsqueeze(2) # 形状: (batch_size, 1, 1, src_seq_len)
    # 目标序列掩码 (结合了 look-ahead mask 和 padding mask)
    tgt_mask = subsequent_mask(tgt_seq_len).unsqueeze(0) # 形状: (1, tgt_seq_len, tgt_seq_len)
    # 如果有 padding，还需要将 padding 部分在 tgt_mask 中设置为 False
    # 例如: tgt_padding_mask = (tgt_data != 0).unsqueeze(1).unsqueeze(2) # 形状: (batch_size, 1, 1, tgt_seq_len)
    # tgt_mask = tgt_mask & tgt_padding_mask

    # 前向传播
    output = model(src_data, tgt_data, src_mask, tgt_mask)

    print("Transformer 模型输出形状:", output.shape) # 期望形状: (batch_size, tgt_seq_len, tgt_vocab_size)
    assert output.shape == (batch_size, tgt_seq_len, tgt_vocab_size)
    print("模型测试通过！")

