import networkx as nx
import numpy as np

def graph_node_label2int(G):
    '''将原始图中的节点的字符型标签转化为整型'''

    if type(G) == type(nx.DiGraph()):
        g = nx.from_edgelist([(int(e[0])-1, int(e[1])-1) for e in G.edges()], create_using=nx.DiGraph())
    else:
        g = nx.from_edgelist([(int(e[0])-1, int(e[1])-1) for e in G.edges()])
    return g

def construct_nodes_feature(G):
    '''计算节点的网络结构特征，使其作为节点的特征'''
    num_nodes = G.number_of_nodes()
    nodes = sorted(list(G.nodes()))
    degree_feat = dict(list(nx.degree(G)))          # 节点的度
    cluceo_feat = nx.clustering(G)      # 节点的聚集系数

    # 构建array类型的节点网络特征矩阵
    nodes_feature = np.array([[degree_feat[i], cluceo_feat[i]] for i in nodes])

    return nodes_feature



