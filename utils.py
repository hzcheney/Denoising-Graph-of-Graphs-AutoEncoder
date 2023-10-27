import random
import torch.cuda
from data_process import *
from texttable import Texttable
import scipy.sparse as sp
import logging
import os
from torch_geometric.utils import negative_sampling, train_test_split_edges

random.seed(666)
np.random.seed(666)



class GraphDatasetGenerator(object):
    def __init__(self, args):
        self.modular_file = args.modular_file
        self.ddi_file = args.ddi_file
        self._load_data(args)
        self.num_nodes = get_num_nodes()

        self._split_data(args)
        self._read_neg(args)

    def _load_data(self, args):
        if args.train_type == "se":
            self.output, self.edges, self.edges_attr, self.se_name = load_data(
                self.modular_file, self.ddi_file, "onehot"
            )
        elif args.train_type == "DB" or args.train_type == "CCI":
            self.output, self.edges = load_data(
                self.modular_file, self.ddi_file, "onehot"
            )
        if torch.cuda.is_available():
            for key in list(self.output.keys()):
                self.output[key][0] = self.output[key][0].to(args.device)
                self.output[key][1] = self.output[key][1].to(args.device)
                self.output[key][2] = self.output[key][2].to(args.device)
                self.output[key][4] = self.output[key][4].to(args.device)
        self.num_atom_features = len(get_atom_features())
        print("atom_features: {}".format(self.num_atom_features))
        self.num_edges = self.edges.size(1) // 2
        train_num = int(self.num_edges * args.train_ratio)
        val_num = int(self.num_edges * args.val_ratio)
        test_num = int(self.num_edges * args.test_ratio)
        self.nums = [train_num, val_num, test_num]

    def _split_data(self, args):
        if args.train_type == "se":
            (
                self.train_edges,
                self.train_edges_attr,
                self.val_edges,
                self.val_edges_attr,
                self.test_edges,
                self.test_edges_attr,
            ) = split_data(self.edges, self.nums, self.se_name)

            self.train_name = self.train_edges_attr
            self.val_name = self.val_edges_attr
            self.test_name = self.test_edges_attr

        elif args.train_type == "DB" or args.train_type == "CCI":
            self.train_edges, self.val_edges, self.test_edges = split_data(
                self.edges, self.nums
            )

        if torch.cuda.is_available():
            self.train_edges = self.train_edges.to(args.device)
            self.val_edges = self.val_edges.to(args.device)
            self.test_edges = self.test_edges.to(args.device)

    def _genarate_neg(self):
        pass

    def _read_neg(self, args):
        # neg_edges = None
        if args.train_type == "se":
            (
                self.neg_train_edges,
                self.neg_val_edges,
                self.neg_test_edges,
            ) = read_negative()
        elif args.train_type == "CCI":
            neg_edges = load_cci(args.neg_ddi_file, list(self.output.keys()))
        elif args.train_type == "DB":
            neg_edges = load_ddi(args.neg_ddi_file, list(self.output.keys()))

        num_edges = neg_edges.size(1) // 2
        train_num = int(num_edges * args.train_ratio)
        val_num = int(num_edges * args.val_ratio)
        test_num = int(num_edges * args.test_ratio)
        nums = [train_num, val_num, test_num]
        self.neg_train_edges, self.neg_val_edges, self.neg_test_edges = split_data(
            neg_edges, nums
        )
        if torch.cuda.is_available():
            self.neg_train_edges = self.neg_train_edges.to(args.device)
            self.neg_val_edges = self.neg_val_edges.to(args.device)
            self.neg_test_edges = self.neg_test_edges.to(args.device)

    def _load(self, args):
        self.num_atom_features = len(get_atom_features())

        self.output, self.edges = load_data(self.modular_file, self.ddi_file, "onehot")
        for key in list(self.output.keys()):
            self.output[key][0] = self.output[key][0].to(args.device)
            self.output[key][1] = self.output[key][1].to(args.device)
            self.output[key][2] = self.output[key][2].to(args.device)
            self.output[key][3] = self.output[key][3].to(args.device)
        self.data = Data(edge_index=self.edges)
        data = train_test_split_edges(
            self.data, val_ratio=args.val_ratio, test_ratio=args.test_ratio
        )
        self.train_edges = data.train_pos_edge_index.to(args.device)
        self.val_edges = data.val_pos_edge_index.to(args.device)
        self.test_edges = data.test_pos_edge_index.to(args.device)

        self.neg_train_edges = negative_sampling(
            self.train_edges, num_nodes=len(self.output)
        )
        self.neg_test_edges = data.test_neg_edge_index.to(args.device)
        self.neg_val_edges = data.val_neg_edge_index.to(args.device)
