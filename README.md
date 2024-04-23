# KGïA: Leveraging Disease-Specific Topologies and Counterfactual Relationships in Knowledge Graphs for Inductive Reasoning in Drug Repurposing

## How to Run KGïA

### 1. Data Splitting
This code generates the validation (`valid.txt.`) and test splits (`text.txt`) for semi-inductive and transductive settings given a train split (`train.txt`).

```bash
python split_KG.py -kg_filepath 'biomedical_KG.txt' -train_filepath 'path/to/your/train_split/train.txt'
```

### 2. Data Augmentation
#### 2.1 Augment Train Set
Generate augmentation edges for train.txt using different settings of treatment/manipulation, applicable for both transductive and semi-inductive settings. This step introduces counterfactual links for triples with a TREAT relation. You can modify the treatment settings as needed.

```bash
python main.py --dataset pubmed --metric auc --alpha 1 --beta 1 --gamma 30 --lr 0.1 --embraw mvgrl --t kcore --neg_rate 40 --jk_mode mean --batch_size 12000 --epochs 200 --patience 50 --trail 20
```
Make sure `edges_f_t0.npy`, `edges_f_t1.npy`, and `int_to_entity.pkl` are in a directory named after your chosen treatment.
```bash
python augment.py -treatment kcore
```

### 3. Prediction
Ensure that your data splits are stored under ULTRA/datasets.

To conduct a case study in which you wish to annotate each triple in your test file with its score and rank, include the argument `-this_is_a_case_study`

#### 3.1 Zero-Shot Semi-Inductive Prediction
```bash
python script/run.py -c config/inductive/inference.yaml --dataset GPKG_SInd --version 'na' --epochs 0 --bpe null --gpus [0] --ckpt /home/cxo147/ULTRA_PDR/ckpts/ultra_50g.pth
```
#### 3.2 Zero-Shot Transductive Prediction
```bash
python script/run.py -c config/transductive/inference.yaml --dataset GPKG_T --epochs 0 --bpe null --gpus [0] --ckpt /home/cxo147/ULTRA_PDR/ckpts/ultra_50g.pth
```
#### 3.3 Distributed Fine-Tuning for Semi-Inductive Prediction
```bash
python -m torch.distributed.launch --nproc_per_node=4 script/run.py -c config/inductive/inference.yaml --dataset GPKG_FI --version 'na' --epochs 10 --bpe 100 --gpus [0,1,2,3] --ckpt /home/cxo147/ULTRA_PDR/ckpts/ultra_50g.pth
```
