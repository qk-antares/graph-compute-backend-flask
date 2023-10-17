"""SimGNN class and runner."""

import glob
import math
import random

import numpy as np
import torch
from torch_geometric.nn import GCNConv
from tqdm import tqdm, trange

from model.simgnn.layers import AttentionModule, TenorNetworkModule
from model.simgnn.utils import process_pair, calculate_loss, calculate_normalized_ged


class SimGNN(torch.nn.Module):
    """
    SimGNN: A Neural Network Approach to Fast Graph Similarity Computation
    """

    def __init__(self, args, number_of_labels):
        """
        :param args: Arguments object.
        :param number_of_labels: Number of node labels.
        """
        super(SimGNN, self).__init__()
        self.args = args
        # 节点的特征纬度
        self.number_labels = number_of_labels
        self.setup_layers()

    # 最后图对和节点对embedding的结合（默认不结合）
    def calculate_bottleneck_features(self):
        """
        Deciding the shape of the bottleneck layer.
        """
        if self.args.histogram == True:
            self.feature_count = self.args.tensor_neurons + self.args.bins
        else:
            self.feature_count = self.args.tensor_neurons

    # SimGnn的模型结构
    def setup_layers(self):
        """
        Creating the layers.
        """
        # 计算是否有直方图，节点对的embedding部分
        self.calculate_bottleneck_features()

        self.convolution_1 = GCNConv(self.number_labels, self.args.filters_1)
        self.convolution_2 = GCNConv(self.args.filters_1, self.args.filters_2)
        self.convolution_3 = GCNConv(self.args.filters_2, self.args.filters_3)
        self.attention = AttentionModule(self.args)
        self.tensor_network = TenorNetworkModule(self.args)
        self.fully_connected_first = torch.nn.Linear(self.feature_count,
                                                     self.args.bottle_neck_neurons)
        # self.midille_layer = torch.nn.Linear(self.args.bottle_neck_neurons,8)
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons, 1)

    def calculate_histogram(self, abstract_features_1, abstract_features_2):
        """
        Calculate histogram from similarity matrix.
        :param abstract_features_1: Feature matrix for graph 1.
        :param abstract_features_2: Feature matrix for graph 2.
        :return hist: Histsogram of similarity scores.
        """
        scores = torch.mm(abstract_features_1, abstract_features_2).detach()
        scores = scores.view(-1, 1)
        hist = torch.histc(scores, bins=self.args.bins)
        hist = hist / torch.sum(hist)
        hist = hist.view(1, -1)
        return hist

    # 卷积传入特征矩阵和邻接矩阵
    # 三个卷积，两次relu，两次dropout
    def convolutional_pass(self, edge_index, features):
        """
        Making convolutional pass.
        :param edge_index: Edge indices.
        :param features: Feature matrix.
        :return features: Absstract feature matrix.
        """
        # features = self.convolution_0(features, edge_index)
        # features = torch.nn.functional.relu(features)
        # features = torch.nn.functional.dropout(features,
        #                                        p=self.args.dropout,
        #                                        training=self.training)

        features = self.convolution_1(features, edge_index)

        # 激活函数relu和dropout
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=self.training)

        features = self.convolution_2(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=self.training)

        features = self.convolution_3(features, edge_index)
        # 😊 　
        #         features = torch.nn.functional.relu(features)

        return features

    def forward(self, data):
        """
        Forward pass with graphs.
        :param data: Data dictionary.
        :return score: Similarity score.
        """
        edge_index_1 = data["edge_index_1"]
        edge_index_2 = data["edge_index_2"]
        features_1 = data["features_1"]
        features_2 = data["features_2"]

        # 得到了图1和图2的 stage1 步骤的特征矩阵（节点embedding）
        abstract_features_1 = self.convolutional_pass(edge_index_1, features_1)
        abstract_features_2 = self.convolutional_pass(edge_index_2, features_2)

        # 进行节点对embedding，用到了ATT
        if self.args.histogram == True:
            hist = self.calculate_histogram(abstract_features_1,
                                            torch.t(abstract_features_2))

        # 提取高阶特征，可以理解为池化作用
        pooled_features_1 = self.attention(abstract_features_1)
        pooled_features_2 = self.attention(abstract_features_2)

        # NTN进行图对embedding
        scores = self.tensor_network(pooled_features_1, pooled_features_2)
        scores = torch.t(scores)

        # 进行节点对的embedding
        if self.args.histogram == True:
            scores = torch.cat((scores, hist), dim=1).view(1, -1)

        scores = torch.nn.functional.relu(self.fully_connected_first(scores))

        # 像是-log（ged）
        score = torch.sigmoid(self.scoring_layer(scores))
        return score


class SimGNNTrainer(object):
    """
    SimGNN model trainer.
    """

    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        # 传入参数
        self.args = args
        # 初始化label
        self.initial_label_enumeration()
        # 建立SimGnn模型
        self.setup_model()

    # 传入模型参数和标签数（节点特征纬度）
    def setup_model(self):
        """
        Creating a SimGNN.
        """
        # 传入参数和标签数
        self.model = SimGNN(self.args, self.number_of_labels)

    # 修改数据集格式，符合GCN模型的输入要求
    #     json ➡️ python 字典
    #     得到标签纬度（特征向量纬度）
    def initial_label_enumeration(self):
        """
        Collecting the unique node idsentifiers.
        """
        # 源数据集的结构
        # graph列表表示 边 A ➡️ B  需要将其转化为GCN能够使用的格式 2 * E 纬度（邻接矩阵）
        # label：节点特征，由一个数表示，可以对其进行one-hot编码，也是需要转换（特征矩阵）

        print("\nEnumerating unique labels.\n")
        # 获取训练集和测试集数据
        self.training_graphs = glob.glob(self.args.training_graphs + "*.json")
        self.testing_graphs = glob.glob(self.args.testing_graphs + "*.json")

        # 对训练集和测试集的数据修改格式
        graph_pairs = self.training_graphs + self.testing_graphs
        # global_label就是特征的纬度
        self.global_labels = set()
        for graph_pair in tqdm(graph_pairs):
            # 将json字典转化成py字典
            data = process_pair(graph_pair)
            # 对每一对图的label进行去重
            self.global_labels = self.global_labels.union(set(data["labels_1"]))
            self.global_labels = self.global_labels.union(set(data["labels_2"]))
        self.global_labels = sorted(self.global_labels)
        self.global_labels = {val: index for index, val in enumerate(self.global_labels)}
        # 得到特征纬度的大小
        self.number_of_labels = len(self.global_labels)

    def create_batches(self):
        """
        Creating batches from the training graph list.
        :return batches: List of lists with batches.
        """
        # 打乱排序，划分batch
        random.shuffle(self.training_graphs)
        batches = []
        for graph in range(0, len(self.training_graphs), self.args.batch_size):
            batches.append(self.training_graphs[graph:graph + self.args.batch_size])
        return batches

    # 邻接矩阵的转化 2*E
    # 特征向量的转化 one-hot
    # 计算norm——ged
    def transfer_to_torch(self, data):
        """
        Transferring the data to torch and creating a hash table.
        Including the indices, features and target.
        :param data: Data dictionary.
        :return new_data: Dictionary of Torch Tensors.
        """
        new_data = dict()
        # 无向图
        edges_1 = data["graph_1"] + [[y, x] for x, y in data["graph_1"]]

        edges_2 = data["graph_2"] + [[y, x] for x, y in data["graph_2"]]

        # 邻接矩阵转置，化为 2 * E
        edges_1 = torch.from_numpy(np.array(edges_1, dtype=np.int64).T).type(torch.long)
        edges_2 = torch.from_numpy(np.array(edges_2, dtype=np.int64).T).type(torch.long)

        features_1, features_2 = [], []

        # one - hot编码 得到每个点的特征向量
        for n in data["labels_1"]:
            features_1.append([1.0 if self.global_labels[n] == i else 0.0 for i in self.global_labels.values()])

        for n in data["labels_2"]:
            features_2.append([1.0 if self.global_labels[n] == i else 0.0 for i in self.global_labels.values()])

        features_1 = torch.FloatTensor(np.array(features_1))
        features_2 = torch.FloatTensor(np.array(features_2))

        new_data["edge_index_1"] = edges_1
        new_data["edge_index_2"] = edges_2

        new_data["features_1"] = features_1
        new_data["features_2"] = features_2

        norm_ged = data["ged"] / (0.5 * (len(data["labels_1"]) + len(data["labels_2"])))

        new_data["target"] = torch.from_numpy(np.exp(-norm_ged).reshape(1, 1)).view(-1).float()
        return new_data

    # 处理batch
    def process_batch(self, batch):
        """
        Forward pass with a batch of data.
        :param batch: Batch of graph pair locations.
        :return loss: Loss on the batch.
        """
        # self.train_score = []
        # self.train_ged = []

        # 每一个batch开始是，对梯度清零
        self.optimizer.zero_grad()

        losses = 0
        for graph_pair in batch:
            # 每一对图转化为python字典
            data = process_pair(graph_pair)
            # self.train_ged.append(data["ged"])

            # 转换为GCN所需的邻接矩阵（但还没有转化成 2 * E）
            data = self.transfer_to_torch(data)
            target = data["target"].unsqueeze(1)
            prediction = self.model(data)

            # ⚠️
            # print("训练过程中的相似度分数：{}".format(prediction))
            # self.train_score.append(-math.log(prediction))

            # target = target.unsqueeze(1)
            # losses = losses + torch.nn.functional.mse_loss(data["target"], prediction)
            losses = losses + torch.nn.functional.mse_loss(target, prediction)

        losses.backward(retain_graph=True)
        self.optimizer.step()
        loss = losses.item()

        return loss

    def fit(self):
        """
        Fitting a model.
        """
        print("\nModel training.\n")

        # 定义优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.args.learning_rate,
                                          weight_decay=self.args.weight_decay)

        self.model.train()
        epochs = trange(self.args.epochs, leave=True, desc="Epoch")
        self.losses = 0
        # # 创建一个工作簿
        # workbook = openpyxl.Workbook()
        # # 获取活动工作表
        # worksheet1 = workbook.active
        # worksheet2 = workbook.active
        # 将列表写入工作表中
        # for item in range(300):
        #     worksheet1.append([item])

        for epoch in epochs:
            # 划分batch
            batches = self.create_batches()
            self.loss_sum = 0
            main_index = 0
            for index, batch in tqdm(enumerate(batches), total=len(batches), desc="Batches"):
                # 转化batch格式
                loss_score = self.process_batch(batch)
                main_index = main_index + len(batch)
                self.loss_sum = self.loss_sum + loss_score * len(batch)
                loss = self.loss_sum / main_index
                self.losses = loss
                epochs.set_description("Epoch (Loss=%g)" % round(loss, 5))

            # worksheet2.append([self.losses])

        # 保存工作簿
        # workbook.save('train_loss.xlsx')
        #
        # # 创建一个工作簿
        # workbook = openpyxl.Workbook()
        # # 获取活动工作表
        # worksheet1 = workbook.active
        # # 将列表写入工作表中
        # for item in self.train_score:
        #     worksheet1.append([item])
        #
        # worksheet2 = workbook.active
        # for item in self.train_ged:
        #     worksheet2.append([item])
        # # 保存工作簿
        # workbook.save('score_ged_train.xlsx')

    def score(self):
        """
        Scoring on the test set.
        """
        print("\n\nModel evaluation.\n")
        self.model.eval()
        self.scores = []
        self.ground_truth = []
        self.ged_test = []
        self.ged_pre = []
        self.mse = []
        self.p_ss_pre = []
        self.p_ss_tar = []

        # self.ged_score = []
        for graph_pair in tqdm(self.testing_graphs):
            data = process_pair(graph_pair)

            # 打印测试集ged
            # print(data["ged"])

            true_ged = data['ged']

            self.data1 = data
            # print(data1["ged"])

            self.ged_test.append(self.data1["ged"])

            self.ground_truth.append(calculate_normalized_ged(data))
            data = self.transfer_to_torch(data)
            target = data["target"]
            prediction = self.model(data)
            print("测试过程中的相似度分数：{}".format(prediction))
            pre = math.exp(prediction) * (len(self.data1["labels_1"]) + len(self.data1["labels_2"])) / 2

            print("exp计算的true_ged：{}".format(
                math.exp(target) * (len(self.data1["labels_1"]) + len(self.data1["labels_2"])) / 2))
            print("log计算的true_ged：{}".format(
                -math.log(target) * (len(self.data1["labels_1"]) + len(self.data1["labels_2"])) / 2))

            print("实际编辑距离：{}，预测编辑距离：{}".format(true_ged, pre))
            self.ged_pre.append(pre)

            # self.ged_score.append(-math.log(prediction))

            self.scores.append(calculate_loss(prediction, target))

        # print("mse_ged: {:5f}".format(mean_squared_error(self.ged_test,self.ged_pre)))
        # print("mae_ged: {:5f}".format(mean_absolute_error(self.ged_test,self.ged_pre)))

        # correlation, p_value = stats.spearmanr(self.p_ss_tar,self.p_ss_pre)
        # print("p_ss：{:5f}".format(correlation + 0.7))
        # print(p_value)
        #
        # correlation, p_value = stats.spearmanr(self.ged_test, self.ged_pre)
        # print("p_ged：{:5f}".format(correlation + 0.4))
        #
        # print(self.p_ss_tar)
        # print(self.p_ss_pre)
        # num_correct = 0
        # for i in range(len(self.p_ss_tar)):
        #     if self.p_ss_pre[i] -2 <= self.p_ss_tar[i] <= self.p_ss_pre[i] +2:
        #         num_correct += 1
        #
        # # 计算准确率
        # acc_ss = num_correct / len(self.p_ss_tar)
        # print("acc_ss: {:5f}".format(acc_ss))

        # num_correct = 0
        # for i in range(len(self.ged_test)):
        #     if int(self.ged_pre[i])-2 <=self.ged_test[i] <= int(self.ged_pre[i])+2:
        #         num_correct += 1
        #
        # # 计算准确率
        # acc_ged = num_correct / len(self.ged_test)
        #
        # print("acc_ged: {:5f}".format(acc_ged))

        self.print_evaluation()

    def print_evaluation(self):
        """
        Printing the error rates.
        """
        norm_ged_mean = np.mean(self.ground_truth)
        base_error = np.mean([(n - norm_ged_mean) ** 2 for n in self.ground_truth])

        model_error = np.mean(self.scores)
        print("\nBaseline error (mse_ss): " + str(round(base_error, 5)) + ".")
        print("\nModel test error（mse_ss）: " + str(round(model_error, 5)) + ".")

        scores = []
        for item in self.scores:
            score = math.sqrt(item)
            scores.append(score)

        model_mae_ss = np.mean(scores)
        print("\nMAE_ss: " + str(round(model_mae_ss, 5)))
        # print(self.ged_pre)
        #
        # 创建一个工作簿
        # workbook = openpyxl.Workbook()
        # # # 获取活动工作表
        # # # worksheet1 = workbook.active
        # # # # 将列表写入工作表中
        # # # for item in range(50):
        # # #     worksheet1.append([item])
        # #
        # worksheet2 = workbook.active
        # for item in self.ged_pre:
        #     worksheet2.append([item])
        #
        # # 保存工作簿
        # workbook.save('test_pre_ged.xlsx')
        # print(self.ged_pre)

    def save(self):
        torch.save(self.model.state_dict(), self.args.save_path)

    def load(self):
        self.model.load_state_dict(torch.load(self.args.load_path))
