import numpy as np
import torch
import random
from sklearn import metrics
from tqdm import tqdm
from model import DVGGA
from utils import GraphDatasetGenerator, get_num_nodes
from metrics import get_roc_score, eval_threshold, softmax_array
import matplotlib.pyplot as plt


class DVGGALearning(object):
    def __init__(self, args):
        super(DVGGALearning, self).__init__()
        self.args = args
        self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        self.dataset_generator = GraphDatasetGenerator(self.args)
        self.train_ratio = self.args.train_ratio
        self.val_ratio = self.args.val_ratio
        self.test_ratio = self.args.test_ratio
        self.D_criterion = torch.nn.BCEWithLogitsLoss()
        # self.logger = create_logger(name="train", save_dir=args.model_path, quiet=False)
        self.seed = 666

    def seed_everything(self, seed=666):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    def _set_model(self):
        self.model = DVGGA(
            self.args,
            self.dataset_generator.num_atom_features,
            len(self.dataset_generator.output),
        ).to(self.args.device)

    def fit(self):
        self.seed_everything(self.seed)
        self._set_model()
        # input for model
        # logger = self.logger
        # logger.info('Create model and optimizer')
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.learning_rate, weight_decay=5e-4
        )
        milestones = [300, 600]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones, gamma=0.1, last_epoch=-1
        )
        torch.nn.init.xavier_normal_(self.model.emb.weight)
        for epoch in tqdm(range(self.args.num_epoch)):
            self.model.train()
            optimizer.zero_grad()
            in_data = [
                self.dataset_generator.output,
                self.dataset_generator.train_edges,
                self.dataset_generator.neg_train_edges,
            ]

            pre_loss, positive_penalty, pos_pred, neg_pred = self.model(in_data)
            loss = pre_loss + self.args.beta2 * positive_penalty
            loss.backward()
            optimizer.step()
            scheduler.step()

            if epoch % 50 == 0:
                result = self.val()
                log = ""
                m_s = ["AUC", "AP", "ACC", "F1", "Recall"]
                for index, metric in enumerate(result):
                    log += "Epoch: {:03d}, {}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f} \n".format(
                        epoch + 1, m_s[index], metric[0], metric[1], metric[2]
                    )
                print(log)

        # print(self.test())

    @torch.no_grad()
    def test(self):
        res = dict()
        self.model.eval()
        auc, ap, acc, f1, recall, _ = get_roc_score(
            self.model,
            self.dataset_generator.output,
            self.dataset_generator.train_edges,
            self.dataset_generator.neg_train_edges,
        )
        return [auc, ap, acc, f1, recall]

    def val(self):
        auc, ap, acc, f1, recall, loss = get_roc_score(
            self.model,
            self.dataset_generator.output,
            self.dataset_generator.train_edges,
            self.dataset_generator.neg_train_edges,
        )
        val_auc, val_ap, val_acc, val_f1, val_recall, val_loss = get_roc_score(
            self.model,
            self.dataset_generator.output,
            self.dataset_generator.val_edges,
            self.dataset_generator.neg_val_edges,
        )
        test_auc, test_ap, test_acc, test_f1, test_recall, test_loss = get_roc_score(
            self.model,
            self.dataset_generator.output,
            self.dataset_generator.test_edges,
            self.dataset_generator.neg_test_edges,
        )
        auc_result = [auc, val_auc, test_auc]
        ap_result = [ap, val_ap, test_ap]
        acc_result = [acc, val_acc, test_acc]
        f1_result = [f1, val_f1, test_f1]
        recall_result = [recall, val_recall, test_recall]
        return [auc_result, ap_result, acc_result, f1_result, recall_result]
