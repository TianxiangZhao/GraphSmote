# GraphSmote
Pytorch implementation of paper ['GraphSMOTE: Imbalanced Node Classification on Graphs with Graph Neural Networks']() on WSDM2021

## Dependencies
### CPU
- python3
- ipdb
- pytorch1.0
- network 2.4
- dgl-0.4.3

## Dataset
Three processed datasets are published, including BlogCatalog, Cora and Twitter. For downloading them, please [click]().
Please unzip them in the 'data' folder.

## Configurations
### Baselines
We provide five baselines in this code. They can be configured via the '--setting' arguments. Please refer to the 'get_parser()' function in utils.py.
- Oringinal model: Vanilla backbone models. '--setting='no''
- Over-sampling in raw input domain: Repeat nodes in the minority classes. '--setting='upsampling''
- Reweight: Give samples from minority classes a larger weight when calculating the loss. '--setting='reweight''
- SMOTE: Oversampling in raw input domain via SMOTE. '--setting='smote''
- Embed-SMOTE: Perform SMOTE in the intermediate embedding domain. '--setting='embed_up''

### Architectures
We provide two base architectures, GCN and GraphSage. The default one is GraphSage, and can be set via '--model'.

## GraphSMOTE
### Train
- Pretrain

- Finetune

### Test


## Citation


If any problems occurs via running this code, please contact us at tkz5084@psu.edu.

Thank you!


