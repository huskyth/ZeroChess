import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

steps = np.linspace(0, np.pi * 2, 100, dtype=np.float32)
input_x = np.sin(steps)
target_y = np.cos(steps)
plt.plot(steps, input_x, 'b-', label='input:sin')
plt.plot(steps, target_y, 'r-', label='target:cos')

plt.legend(loc='best')
plt.show()


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )

        self.out = nn.Linear(hidden_size, 1)

    def forward(self, input, hidden):
        output, hidden = self.rnn(input, hidden)
        output = self.out(output)
        return output, hidden

    def initHidden(self):
        hidden = torch.randn(1, self.hidden_size)
        return hidden


rnn = RNN(input_size=1, hidden_size=20)

hidden = rnn.initHidden()

optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)
loss_func = nn.MSELoss()

plt.figure(1, figsize=(12, 5))
plt.ion()  # 开启交互模式

loss_list = []
for step in range(800):
    start, end = step * np.pi, (step + 1) * np.pi
    steps = np.linspace(start, end, 100, dtype=np.float32)
    x_np = np.sin(steps)
    y_np = np.cos(steps)
    # (100, 1) 不加batch_size
    x = torch.from_numpy(x_np).unsqueeze(-1)
    y = torch.from_numpy(y_np).unsqueeze(-1)

    y_predict, hidden = rnn(x, hidden)
    hidden = hidden.data  # 重新包装数据，断掉连接，不然会报错
    loss = loss_func(y_predict, y)
    optimizer.zero_grad()  # 梯度清零
    loss.backward()  # 反向传播
    optimizer.step()  # 梯度下降
    loss_list.append(loss.item())
    if step % 10 == 0 or step % 10 == 1:
        plt.plot(steps, y_np.flatten(), 'r-')
        plt.plot(steps, y_predict.data.numpy().flatten(), 'b-')
        plt.draw()
        plt.pause(0.05)

plt.ioff()
plt.show()

plt.plot(loss_list)
