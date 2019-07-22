import time
import torch
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid

class Graphsage(torch.nn.Module):
    def __init__(self, input1):
        super(Graphsage, self).__init__()
        self.conv1 = SAGEConv(dataset.num_node_features, input1)
        self.conv2 = SAGEConv(input1, dataset.num_classes)
        # self.conv3 = SAGEConv(imput2, dataset.num_classes)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        # x = F.relu(x)
        # x = self.conv3(x, edge_index)

        return F.log_softmax(x, dim=1)


if __name__=="__main__":

    # 加载数据集，Planetoid加载的是数据集，具体数据在.data属性中。
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 声明模型：数据集的节点特征数目，第一层输出，既第二层得到的节点特征数目，最后输出的特征数目，也就是分类情况
    data = data.to(device)
    model = Graphsage(32).to(device)

    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # 分批次训练模型
    model.train()
    batch_time = 0
    for epoch in range(200):
        start_time = time.time()
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        end_time = time.time()
        batch_time += 1
        print("第{}次训练的损失为：{}".format(batch_time, loss.data.item()))

    # 训练结果的测试
    model.eval()
    _, pred = model(data).max(dim=1)
    correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / data.test_mask.sum().item()
    print('Accuracy: {:.4f}'.format(acc))
