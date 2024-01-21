import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score

# 数据预处理函数
def read_data(file, label):
    sequences = {}
    with open(file, 'r') as f:
        for line in f:
            user_id, op_id, _ = line.strip().split('\t')  # 忽略时间戳
            if user_id not in sequences:
                sequences[user_id] = []
            sequences[user_id].append(int(op_id))
    return [[sequences[user_id], label] for user_id in sequences]


# Transformer 模型类
class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)
        self.linear = nn.Linear(hidden_size, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        attn_output, _ = self.attention(embedded, embedded, embedded)
        output = self.linear(attn_output.mean(dim=0))
        return self.activation(output)

# 训练函数
def train(model, train_data, test_data, epochs=100, learning_rate=0.001):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for seq, label in train_data:
            optimizer.zero_grad()
            # 更新序列张量的转置方式
            seq_tensor = torch.LongTensor(seq).unsqueeze(0).permute(1, 0)
            label_tensor = torch.FloatTensor([label]).squeeze()
            output = model(seq_tensor).squeeze()
            loss = criterion(output, label_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_data)}')
        if epoch % 10 == 0:
            accuracy, recall = evaluate(model, test_data)
            print(f'Accuracy: {accuracy}, Recall: {recall}')





# 评估函数
def evaluate(model, test_data):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for seq, label in test_data:
            seq_tensor = torch.LongTensor(seq).T
            output = model(seq_tensor)
            prediction = output.round().item()
            y_true.append(label)
            y_pred.append(prediction)
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return accuracy, recall

def main():
    # 读取数据
    positive_data = read_data('data/positive.txt', 0)
    negative_data = read_data('data/negative.txt', 1)
    data = positive_data + negative_data

    # 划分训练集和测试集
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # 确定操作 ID 的最大值
    max_op_id = max(max(seq) for seq, _ in data)  # 正确计算最大操作 ID
    # 创建模型实例
    model = TransformerModel(input_size=max_op_id + 1, hidden_size=64, num_heads=2)

    # 训练和评估模型
    train(model, train_data, test_data)
    # accuracy, recall = evaluate(model, test_data)
    # print(f'Accuracy: {accuracy}, Recall: {recall}')


if __name__ == '__main__':
    main()
