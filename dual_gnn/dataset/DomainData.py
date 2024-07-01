import os.path as osp
import torch
import numpy as np
import pandas as pd
import networkx as nx
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.io import read_txt_array
from torch_geometric.utils import to_networkx, from_networkx
from scipy.stats import rankdata


class DomainData(InMemoryDataset):
    r"""The protein-protein interaction networks from the `"Predicting
    Multicellular Function through Multi-layer Tissue Networks"
    <https://arxiv.org/abs/1707.04638>`_ paper, containing positional gene
    sets, motif gene sets and immunological signatures as features (50 in
    total) and gene ontology sets as labels (121 in total).

    Args:
        root (string): Root directory where the dataset should be saved.
        split (string): If :obj:`"train"`, loads the training dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"test"`, loads the test dataset. (default: :obj:`"train"`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """
    def __init__(self,
                 root,  # Root directory where the dataset should be saved.
                 name,
                 domain,
                 lblToRemove1,
                 lblToRemove2,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        self.name = name  # dataset name
        self.domain = domain  # two types: source or target, for different procedures
        self.lblToRemove1 = lblToRemove1
        self.lblToRemove2 = lblToRemove2
        # self.root = root  # data/acmv9
        super(DomainData, self).__init__(root, transform, pre_transform, pre_filter)
        self.process()  # reset the dataset every time
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["docs.txt", "edgelist.txt", "labels.txt"]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        edge_path = osp.join(self.raw_dir, '{}_edgelist.txt'.format(self.name))
        edge_index = read_txt_array(edge_path, sep=',', dtype=torch.long).t()

        docs_path = osp.join(self.raw_dir, '{}_docs.txt'.format(self.name))
        f = open(docs_path, 'rb')
        content_list = []
        for line in f.readlines():
            line = str(line, encoding="utf-8")
            content_list.append(line.split(","))
        x = np.array(content_list, dtype=float)
        x = torch.from_numpy(x).to(torch.float)

        label_path = osp.join(self.raw_dir, '{}_labels.txt'.format(self.name))
        f = open(label_path, 'rb')
        content_list = []
        for line in f.readlines():
            line = str(line, encoding="utf-8")
            line = line.replace("\r", "").replace("\n", "")
            content_list.append(line)
        y = np.array(content_list, dtype=int)
        y_df = pd.DataFrame(y)  # for delete label use

        data = Data(edge_index=edge_index, x=x, y=y)
        # print("Original PyG data: ")
        # print(data)
        # print("Original data label distribution:")
        # print(y_df.value_counts())
        # print(y_df.value_counts(normalize=True))

        # label_to_remove = [3, 4, 5]  # for 6 categories
        # label_to_remove = [3, 4]  # for 5 categories
        label_to_remove = [self.lblToRemove1, self.lblToRemove2]  # for 5 categories

        # modify source domain data: delete nodes and edges with certain labels
        if self.domain == 'source':
            data_nx = to_networkx(data=data, node_attrs=["x"], to_undirected=False)
            # print("data_nx: ", data_nx)
            # print("data_nx number of nodes: ", data_nx.number_of_nodes())
            data_labels = pd.read_csv(label_path, sep=' ', header=None)
            node_to_remove = []
            for l in label_to_remove:
                node_to_remove = node_to_remove + (data_labels[(data_labels[0] == l)].index.tolist())
            # print("length of node_to_remove: ", len(node_to_remove))
            data_nx.remove_nodes_from(node_to_remove)
            # print("data_nx after removing: ", data_nx)
            y_df_new = y_df.drop(node_to_remove)
            # print("y after removing two labels: ", y_df_new)
            y_array_new = y_df_new[0].to_numpy()
            y_array_new = rankdata(y_array_new, method='dense') - 1
            y = torch.from_numpy(y_array_new).to(torch.int64)
            # print("y after removing two labels and re-ranked: ", y)
            # print("new number of nodes after removing: ", data_nx.number_of_nodes())
            data = from_networkx(data_nx)
            data.y = y

        # modify target domain data: combine multiple unseen labels as one label
        if self.domain == 'target':
            y_original = y.copy()
            # print("original y: ", y)
            y[np.any([y_original==self.lblToRemove1, y_original==self.lblToRemove2], axis=0)] = 99
            y = rankdata(y, method='dense') - 1
            y = torch.from_numpy(y).to(torch.int64)
            # print("y re-ranked: ", y)
            data.y = y

        random_node_indices = np.random.permutation(y.shape[0])
        training_size = int(len(random_node_indices) * 0.7)
        val_size = int(len(random_node_indices) * 0.1)
        train_node_indices = random_node_indices[:training_size]
        val_node_indices = random_node_indices[training_size:training_size + val_size]
        test_node_indices = random_node_indices[training_size + val_size:]

        train_masks = torch.zeros([y.shape[0]], dtype=torch.bool)
        train_masks[train_node_indices] = 1
        val_masks = torch.zeros([y.shape[0]], dtype=torch.bool)
        val_masks[val_node_indices] = 1
        test_masks = torch.zeros([y.shape[0]], dtype=torch.bool)
        test_masks[test_node_indices] = 1

        data.train_mask = train_masks
        data.val_mask = val_masks
        data.test_mask = test_masks

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])
