import networkx as nx
import numpy as np
import torch
import json
import csv
import random
from rdkit import Chem
from rdkit.Chem import GraphDescriptors
from torch_geometric.data import Data
from copy import deepcopy
from typing import Union, Tuple, List, Dict
from torch_geometric.utils import to_dense_adj

random.seed(666)

TRAIN_TYPE = "DB"
FEATURES = set()
NUM_NODES = -1
ONLY_ID = False


def get_atom_features():
    return FEATURES


def get_train_type():
    return TRAIN_TYPE


def get_num_nodes():
    return NUM_NODES


def mol_to_nx(mol):
    G = nx.Graph()

    for atom in mol.GetAtoms():
        G.add_node(
            atom.GetIdx(),
            symbol=atom.GetSymbol(),
            formal_charge=atom.GetFormalCharge(),
            implicit_valence=atom.GetImplicitValence(),
            ring_atom=atom.IsInRing(),
            degree=atom.GetDegree(),
            hybridization=atom.GetHybridization(),
        )
    for bond in mol.GetBonds():
        G.add_edge(
            bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_type=bond.GetBondType()
        )
    return G


def load_se_dict(file_name):
    with open(file_name) as f:
        se_dict = json.load(fp=f)
    return se_dict


def process(file_name):
    id_graph_dict = dict()
    id_smiles_dict = dict()
    global FEATURES
    # df = pd.read_csv(file_name, index_col=False, names=['id', 'smiles'], sep='\t')
    # smiles2id = {smiles: idx for smiles, idx in zip(df['smiles'], range(len(df)))}
    # idx2smiles = [''] * len(smiles2id)
    # for smiles, smiles_idx in smiles2id.items():
    #     idx2smiles[smiles_idx] = smiles
    with open(file_name) as f:
        for line in f:
            line = line.rstrip()
            id, smiles = line.split()
            mol = Chem.MolFromSmiles(smiles)
            atoms = mol.GetAtoms()
            FEATURES = FEATURES.union(i.GetSymbol() for i in atoms)
            graph = mol_to_nx(mol)
            id_graph_dict[id] = graph
            id_smiles_dict[id] = smiles
    # with open('/data/hancwang/DDI/code/data/id_graph_dict.json', 'w') as f:
    #     json.dump(id_graph_dict, f)
    return id_graph_dict, id_smiles_dict


def node_feature_process(G, feature_type="onehot"):
    feature = []
    feature_dict = node_feature_dict(feature_type)
    symbols = nx.get_node_attributes(G, "symbol")
    k = sorted(list(symbols.keys()))
    for key in k:
        feature.append(feature_dict[symbols[key]])
    num_nodes = len(k)
    batch_list = [0] * num_nodes
    batch = torch.IntTensor(batch_list)  # int tensor for batch
    return torch.Tensor(feature), batch


def node_feature_dict(type="onehot"):
    symbol_dict = dict()
    keys = list(get_atom_features())
    num_symbols = len(keys)
    if type == "onehot":
        for i in range(len(keys)):
            temp = [0] * num_symbols
            temp[i] = 1
            feature = temp
            symbol_dict[keys[i]] = deepcopy(feature)  # just in case
    elif type == "fixed":
        for i in range(len(keys)):
            temp = [0] * fixed_dim
            if i <= 7:
                temp[i] = 1
            elif 8 <= i <= 14:
                temp[0] = 1
                temp[i % 8 + 1] = 1
            elif 15 <= i <= 20:
                temp[1] = 1
                temp[(i + 3) % 8] = 1
            else:
                temp[2] = 1
                temp[3] = 1
            feature = temp
            symbol_dict[keys[i]] = deepcopy(feature)  # just in case
    return symbol_dict


def edge_preprocess(G):
    edge_weight = []
    edge_1 = []
    edge_2 = []
    bond_types_dict = {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "AROMATIC": 1.5}
    bonds = nx.get_edge_attributes(G, "bond_type")
    edge_index = list(bonds.keys())
    for edge in edge_index:
        edge_weight.append(
            bond_types_dict[str(bonds[edge])]
        )  # change the name of bond to the string to match the dict
        edge_1.append(edge[0])
        edge_2.append(edge[1])
        edge_1.append(edge[1])
        edge_2.append(edge[0])
        edge_weight.append(bond_types_dict[str(bonds[edge])])
    edge_output = [edge_1, edge_2]
    return torch.LongTensor(edge_output), torch.Tensor(edge_weight)


def load_data(modular_file_name, DDI_file_name, feature_type="onehot"):
    # load the graph with the one-hot node feature, output the dict whose key is cid,
    # values are [node feature, edge index and edge weight] (a list).
    global NUM_NODES
    d, id2smiles = process(modular_file_name)
    NUM_NODES = len(d.keys())
    output = dict()
    for ids in d.keys():
        graph = d[ids]
        smiles = id2smiles[ids]
        node_feature, batch = node_feature_process(graph, feature_type)
        edge_index, edge_weight = edge_preprocess(graph)
        # edge_weight = torch.tensor(edge_weight, dtype=torch.int64)
        # adj = torch.squeeze(to_dense_adj(edge_index), 0)
        output[ids] = [node_feature, edge_index, edge_weight, smiles, batch]
        # output[ids] = [node_feature, edge_index, edge_weight]
    if get_train_type() == "DB":
        edges = load_ddi(DDI_file_name, list(output.keys()))
        return output, edges
    elif get_train_type() == "CCI":
        edges = load_cci(DDI_file_name, list(output.keys()))
        return output, edges


def load_cci(file_name, node_ids):
    edges = []
    edges_1 = []
    edges_2 = []
    id_dict = dict()
    for i in range(len(node_ids)):
        new_id = "CID" + "0" * (9 - len(node_ids[i])) + node_ids[i]
        id_dict[new_id] = i
    with open(file_name) as f:
        sreader = csv.reader(f, delimiter=",")
        next(sreader, None)
        for line in sreader:
            edges_1.append(id_dict[line[0]])
            edges_2.append(id_dict[line[1]])
            edges_1.append(id_dict[line[1]])
            edges_2.append(id_dict[line[0]])
        edges.append(edges_1)
        edges.append(edges_2)
    return torch.LongTensor(edges)


def load_ddi(file_name, node_ids):
    edges = []
    edges_1 = []
    edges_2 = []
    id_dict = dict()
    for i in range(len(node_ids)):
        if ONLY_ID:
            new_id = node_ids[i]
        else:
            new_id = "DB" + "0" * (5 - len(node_ids[i])) + node_ids[i]
        id_dict[new_id] = i
    with open(file_name) as f:
        sreader = csv.reader(f, delimiter=",")
        next(sreader, None)
        for line in sreader:
            edges_1.append(id_dict[line[0]])
            edges_2.append(id_dict[line[1]])
            edges_1.append(id_dict[line[1]])
            edges_2.append(id_dict[line[0]])
        edges.append(edges_1)
        edges.append(edges_2)
    return torch.LongTensor(edges)


def new_index(index_1, index_2, index_3):
    train_index = []
    val_index = []
    test_index = []
    for index in index_1:
        train_index.append(2 * index)
        train_index.append(2 * index + 1)
    for index in index_2:
        val_index.append(2 * index)
        val_index.append(2 * index + 1)
    for index in index_3:
        test_index.append(2 * index)
        test_index.append(2 * index + 1)
    return train_index, val_index, test_index


def split_data(edges, nums, edges_attr=None):
    train_num, val_num, test_num = nums
    # edges_attr = np.array(edges_attr)
    total_nums = edges.size(1) // 2
    index_list = []
    for i in range(total_nums):
        index_list.append(i)
    random.shuffle(index_list)
    train_index = index_list[0:train_num]
    val_index = index_list[train_num : train_num + val_num]
    test_index = index_list[train_num + val_num : train_num + val_num + test_num]
    train_index, val_index, test_index = new_index(train_index, val_index, test_index)
    row, col = edges
    train_edges = torch.cat(
        (torch.unsqueeze(row[train_index], 0), torch.unsqueeze(col[train_index], 0)),
        dim=0,
    )
    val_edges = torch.cat(
        (torch.unsqueeze(row[val_index], 0), torch.unsqueeze(col[val_index], 0)), dim=0
    )
    test_edges = torch.cat(
        (torch.unsqueeze(row[test_index], 0), torch.unsqueeze(col[test_index], 0)),
        dim=0,
    )
    if edges_attr is not None:
        edges_attr = np.array(edges_attr)
        train_attr = edges_attr[train_index]
        val_attr = edges_attr[val_index]
        test_attr = edges_attr[test_index]
        return train_edges, train_attr, val_edges, val_attr, test_edges, test_attr
    else:
        return train_edges, val_edges, test_edges


def is_in(edge_pair, edge_attr, edges, attr):
    if len(attr) == 0:
        return False
    row, col = edges
    for i in range(len(row)):
        if (
            edge_pair[0] == row[i] and edge_pair[1] == col[i] and edge_attr == attr[i]
        ) or (
            edge_pair[1] == row[i] and edge_pair[0] == col[i] and edge_attr == attr[i]
        ):
            return True
    return False


def is_member(edge_pair, edge_attr, edge_dict):
    if edge_attr not in edge_dict.keys():
        return False
    edges = edge_dict[edge_attr]
    row, col = edges
    for i in range(len(row)):
        if (edge_pair[0] == row[i] and edge_pair[1] == col[i]) or (
            edge_pair[1] == row[i] and edge_pair[0] == col[i]
        ):
            return True
    return False


def edge_dict(edges, attr):
    out_dict = {}
    for i in range(len(attr)):
        if attr[i] not in out_dict.keys():
            out_dict[attr[i]] = [[edges[0][i]], [edges[1][i]]]
        else:
            out_dict[attr[i]][0].append(edges[0][i])
            out_dict[attr[i]][1].append(edges[1][i])
    return out_dict
