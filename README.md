# Denoising Variational Graph of Graphs Auto-Encoder for Predicting Structured Entity Interactions
Pytorch-geometeric implementation for TKDE'2023 paper: [Denoising Variational Graph of Graphs Auto-Encoder for Predicting Structured Entity Interactions
](https://ieeexplore.ieee.org/document/10192364)
## Dependencies
The repository also contains the following dependencies. Please refer to the respective page for installation instructions.

- python >= 3.7
- [PyTorch](https://pytorch.org/) >= 1.8.0
- [pytorch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/) >= 2.0.0
- numpy
- rdkit
- scikit-learn
- networkx
- texttable

Run following command to install all necessary dependencies.
```
pip install -r requirements.txt
```
## Create virtual environment
You can choose between virtualenv or anaconda, among other environment management tools, to create a virtual environment. 

## Usage example
Run the script below to get results on ZhangDDI dataset.
```
python main.py --second-gcn-dimensions 256 --num_epoch 600 --learning_rate 0.001 --beta2 1 --train_ratio 0.6 --val_ratio 0.2 --test_ratio 0.2 --train_type DB --modular_file data/zhang/id2smiles.txt --ddi_file data/zhang/pos.csv --neg_ddi_file data/zhang/neg.csv 
```

# FAQ
- If you encounter any issues, please don't hesitate to reach out to me or open an issue. 
# Citing
If you find this work is helpful for your research, please consider citing our paper:
```
@ARTICLE{chen2023dvgga,
  author={Chen, Han and Wang, Hanchen and Chen, Hongmei and Zhang, Ying and Zhang, Wenjie and Lin, Xuemin},
  journal={IEEE Transactions on Knowledge and Data Engineering}, 
  title={Denoising Variational Graph of Graphs Auto-Encoder for Predicting Structured Entity Interactions}, 
  year={2023},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/TKDE.2023.3298490}}

```
Some part of this project are build upon [GoGNN](https://github.com/Hanchen-Wang/GoGNN). If you find it helpful for your research, please consider citing this paer:
```
@article{wang2020gognn,
  title={Gognn: Graph of graphs neural network for predicting structured entity interactions},
  author={Wang, Hanchen and Lian, Defu and Zhang, Ying and Qin, Lu and Lin, Xuemin},
  journal={arXiv preprint arXiv:2005.05537},
  year={2020}
}
```