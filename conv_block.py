from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, time_embedding_size):
        super(ConvBlock, self).__init__()

        self.sequence1 = nn.Sequential(
            # 为什么当kernel_size=3 & stride=1 & padding=1时, 卷积层的输出空间尺寸不变呢？
            # 首先, output_size = [(input_size + 2 * padding - kernel_size) / stride] + 1
            # 则height = [(input_size + 2 * 1 - 3) / 1] + 1 = input_size
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),  # 改变通道数, 不改变大小
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

        self.time_embedding_linear = nn.Linear(time_embedding_size, out_channel)
        self.relu = nn.ReLU()

        self.sequence2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),  # 不改变通道数, 不改变大小
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

    def forward(self, hidden_states, time_embedding):
        r"""
        Args:
            hidden_states:  [b, c, h, w]
            time_embedding: [b,time_embedding_size]
        """
        hidden_states = self.sequence1(hidden_states) # [b, output_channels, h, w]
        time_embedding = self.relu(self.time_embedding_linear(time_embedding).view(hidden_states.shape[0], hidden_states.shape[1], 1, 1)) # [b, output_channels, 1, 1]
        # hidden_states + time_embedding会调用广播机制, [b, output_channels, h, w] -> [b, output_channels, h, w]
        return self.sequence2(hidden_states + time_embedding)  