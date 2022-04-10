![](https://img.shields.io/badge/version-1.0.0-blue)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/Fangyin1994/KCL/blob/main/LICENSE)
[![arxiv badge](https://img.shields.io/badge/arxiv-2112.00544-orange)](https://arxiv.org/abs/2112.00544)

[**中文**](https://github.com/ZJU-Fangyin/KCL/blob/main/README_CN.md) | [**English**](https://github.com/ZJU-Fangyin/KCL)     

<p align="center">
    <a href="https://github.com/zjunlp/openue"> <img src="https://raw.githubusercontent.com/zjunlp/openue/master/docs/images/logo_zju_klab.png" width="400"/></a>
</p>

# Molecular Contrastive Learning with Chemical Element Knowledge Graph

This repository is the official implementation of [**KCL**](https://github.com/ZJU-Fangyin/KCL), which is model proposed in a paper: **[Molecular Contrastive Learning with Chemical Element Knowledge Graph](https://arxiv.org/abs/2112.00544)**, accepted by **AAAI 2022** main conference. 


# Contributor
Yin Fang, Qiang Zhang, Haihong Yang, Xiang Zhuang, Shumin Deng, Wen Zhang, Ming Qin, Zhuo Chen, Xiaohui Fan, Huajun Chen


# Brief Introduction
We construct a Chemical Element Knowledge Graph (KG) to summarize microscopic associations between elements and propose a novel **K**nowledge-enhanced **C**ontrastive **L**earning (**KCL**) framework for molecular representation learning. 


## Model
We construct a Chemical Element Knowledge Graph (KG) to summarize microscopic associations between elements and propose a novel Knowledge-enhanced Contrastive Learning (KCL) framework for molecular representation learning. KCL framework consists of three modules. **The first module**, knowledge-guided graph augmentation, augments the original molecular graph based on the Chemical Element KG. **The second module**, knowledge-aware graph representation, extracts molecular representations with a common graph encoder for the original molecular graph and a Knowledge-aware Message Passing Neural Network (KMPNN) to encode complex information in the augmented molecular graph. **The final module** is a contrastive objective, where we maximize agreement between these two views of molecular graphs.

<div align=center><img src="./fig/overview.png" style="zoom:100%;" />
</div>


# Requirements
To run our code, please install dependency packages.
```
python         3.7
torch          1.7.1
dgl            0.6.1
rdkit          2018.09.3
dgllife        0.2.8
pandarallel    1.5.2
numpy          1.20.3
pandas         1.3.1
lmdb           1.2.1
```

# Preparing
We collect 250K unlabeled molecules sampled from the ZINC 15 datasets to pre-train KCL. The raw pre-training data can be found in `data/raw/zinc15_250K_2D.csv`.

We saved pre-train dataset in LMDB, please execute `cd data` and run:

- `python graph_utils.py`

To apply hard negative strategy, please execute `cd data` and run:

- `bash dist.sh`
- `bash cluster.sh`
- `python uni_cluster.py`

<!-- 
# Training
To pre-train KCL, please execute `cd code` and run:

- `bash script/pretrain.sh` -->


# Running

To test on downstream tasks, please execute `cd code` and run:

- `bash script/finetune.sh`

Change the `data_name` command in the bash file to replace different datasets.

You can also specify the `encoder_name`, `training rate`, etc. in this bash file. 

Don't forget change the `encoder_path` and `readout_path` if you change your `encoder_name`! For example:
```
CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --seed 12 \
    --encoder_name GNN \
    --batch_size 64 \
    --predictor_hidden_feats 32 \
    --patience 100 \
    --encoder_path ./dump/Pretrain/gnn-kmpnn-model/GCNNodeEncoder_0910_0900_2000th_epoch.pkl \
    --readout_path ./dump/Pretrain/gnn-kmpnn-model/WeightedSumAndMax_0910_0900_2000th_epoch.pkl \
    --lr 0.001 \
    --predictor nonlinear \
    --eval nonfreeze \
    --data_name Tox21 \
    --split_type random \
    --dump_path ./dump \
    --exp_name KG-finetune-gnn \
    --exp_id tox21
```


# Pre-trained Models

You can download pretrained models here: `/dump/Pretrain/gnn-kmpnn-model`



<!-- ## Results
We verify the effectiveness of KCL under two settings on 8 benchmark datasets from the MoleculeNet: (1) fine-tune protocol (2) linear protocol.

* performance under fine-tune protocol
<div align=center><img src="./fig/fine-tune_protocol.png" width = "800" />
</div>

<br/>

* performance under linear protocol
<div align=center><img src="./fig/linear_protocol.png" height = "250" />
</div> -->


# Papers for the Project & How to Cite
If you use or extend our work, please cite the following paper:
```
@article{fang2021molecular,
  title={Molecular Contrastive Learning with Chemical Element Knowledge Graph},
  author={Fang, Yin and Zhang, Qiang and Yang, Haihong and Zhuang, Xiang and Deng, Shumin and Zhang, Wen and Qin, Ming and Chen, Zhuo and Fan, Xiaohui and Chen, Huajun},
  journal={arXiv preprint arXiv:2112.00544},
  year={2021}
}
```

# Other Matters
As we plan to expand the paper to a journal, we will publish the training procedure together with the follow-up work in June to protect our work. Thanks for your understanding!
