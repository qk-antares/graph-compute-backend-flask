"""Classes for SimGNN modules."""

import torch


# 完成了图的embedding
class AttentionModule(torch.nn.Module):
    """
    SimGNN Attention Module to make a pass on graph.
    """

    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super(AttentionModule, self).__init__()
        self.args = args
        self.setup_weights()
        self.init_parameters()

    # 为第三层卷积输出的大小
    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.args.filters_3,
                                                             self.args.filters_3))

    # 参数初始化，权重参数
    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, embedding):
        """
        Making a forward propagation pass to create a graph level representation.
        :param embedding: Result of the GCN.
        :return representation: A graph level representation vector.
        """

        # 节点embedding = 节点数 * GCN3的纬度
        # weight = GCN3 * GCN3

        # 得到节点数 * GCN3的矩阵 ——> 对每一列求平均值，得到 1 * GCN3的矩阵
        global_context = torch.mean(torch.matmul(embedding, self.weight_matrix), dim=0)

        transformed_global = torch.tanh(global_context)

        # 激活函数sigmoid
        # 将global转化成 GCN3 * 1，得到 节点数 * 1 的矩阵 sigmoid——scores
        sigmoid_scores = torch.sigmoid(torch.mm(embedding, transformed_global.view(-1, 1)))

        # 得到GCN3 * 1的矩阵 representation
        representation = torch.mm(torch.t(embedding), sigmoid_scores)
        return representation


class TenorNetworkModule(torch.nn.Module):
    """
    SimGNN Tensor Network module to calculate similarity vector.
    """

    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super(TenorNetworkModule, self).__init__()
        self.args = args
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """

        # 定义WK
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.args.filters_3,
                                                             self.args.filters_3,
                                                             self.args.tensor_neurons))

        # 定义 V
        self.weight_matrix_block = torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons,
                                                                   2 * self.args.filters_3))

        # 定义 bias
        self.bias = torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons, 1))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.weight_matrix_block)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, embedding_1, embedding_2):
        """
        Making a forward propagation pass to create a similarity vector.
        :param embedding_1: Result of the 1st embedding after attention.
        :param embedding_2: Result of the 2nd embedding after attention.
        :return scores: A similarity score vector.
        """

        # 转置图1的embedding ， -1 表示GCN3 * tensor-neurons
        scoring = torch.mm(torch.t(embedding_1), self.weight_matrix.view(self.args.filters_3, -1))
        scoring = scoring.view(self.args.filters_3, self.args.tensor_neurons)
        scoring = torch.mm(torch.t(scoring), embedding_2)

        combined_representation = torch.cat((embedding_1, embedding_2))
        block_scoring = torch.mm(self.weight_matrix_block, combined_representation)

        scores = torch.nn.functional.relu(scoring + block_scoring + self.bias)
        return scores
