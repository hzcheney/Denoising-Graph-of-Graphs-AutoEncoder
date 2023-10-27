import argparse
import torch


def para_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pooling_ratio", type=float, default=0.5, help="pooling ratio"
    )
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout ratio")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="training ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="validation ratio")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="test ratio")
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="learning rate for training"
    )
    parser.add_argument(
        "--num_epoch", type=int, default=10, help="number of training epoch"
    )
    parser.add_argument(
        "--modular_file",
        type=str,
        default="data/NetBasedDDI/id2smiles.txt",
        help="file store the modulars information",
    )
    parser.add_argument(
        "--ddi_file",
        type=str,
        default="data/NetBasedDDI/NetBasedDDIPositive_id2id.csv",
        help="file store the ddi information",
    )
    parser.add_argument(
        "--neg_ddi_file",
        type=str,
        default="data/NetBasedDDI/NetBasedDDINegative_id2id.csv",
        help="file store the ddi information",
    )
    parser.add_argument(
        "--model_path", type=str, default="./saved_model/", help="saved model path"
    )
    parser.add_argument(
        "--feature_type",
        type=str,
        default="onehot",
        help="the feature type for the atoms in modulars",
    )
    parser.add_argument(
        "--train_type",
        type=str,
        default="DB",
        help="training type of the model, each batch contains fixed edges or a side effect graph",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=1.0,
        help="beta2 value for positive penalty, set a float value >= 0.",
    )
    parser.add_argument(
        "--first-gcn-dimensions",
        type=int,
        default=256,
        help="Filters (neurons) in 1st convolution. Default is 256.",
    )

    parser.add_argument(
        "--second-gcn-dimensions",
        type=int,
        default=128,
        help="Filters (neurons) in 2nd convolution. Default is 128.",
    )

    parser.add_argument("--first-dense-neurons", type=int, default=256, help="Neurons in SAGE aggregator layer. Default is 16.")

    parser.add_argument("--second-dense-neurons", type=int, default=2, help="assignment. Default is 2.")
    parser.add_argument(
        "--vgae_hidden_dimensions", type=int, default=256, help="VGAE hidden dims"
    )
    parser.add_argument(
        "--test_flag", type=bool, default=False, help="whether to load saved model"
    )
    parser.add_argument("--date_time", default="0000", help="Current date and time.")

    parser.add_argument("--device", default="cuda:0", help="Train device")
    args = parser.parse_args()
    args.cuda = True if torch.cuda.is_available() else False

    return args
