# GraphSmote
Pytorch implementation of paper ['GraphSMOTE: Imbalanced Node Classification on Graphs with Graph Neural Networks']() on WSDM2021

## Dependencies
### CPU
- python3
- ipdb
- pytorch1.0
- network 2.4
- scipy
- sklearn

## Dataset
Two processed datasets are published, including BlogCatalog and Cora. For downloading them, please [click](https://drive.google.com/drive/folders/1rfIfRPG7IlzDMAYqQ25HOQmLBCHcECQx?usp=sharing).
Please unzip them in the 'data' folder.

## Configurations

### Architectures
We provide two base architectures, GCN and GraphSage. The default one is GraphSage, and can be set via '--model'.

### Upscale ratios
The default value is 1. If want to make every class balanced instead of using pre-set ratios, please set it to 0 in '--up_scale'.

### Finetuning the decoder
During finetune, set '--setting='newG_cls'' correponds to use pretrained decoder, and set '--setting='recon_newG'' corresponds to also finetune the decoder.

Besides, during finetune, '--opt_new_G' corresponds to update decoder with also classification losses. This option may cause more variance in performance, and usually need more careful hyper-parameter choices.

## GraphSMOTE
Below is an example for the Cora dataset.

### Train
- Pretrain the auto-encoder

<code>python main.py --imbalance --no-cuda --dataset=cora --setting='recon'</code>

Pretrained model can be found in the corresponding checkpoint folder. Rename and set the path to pretrained checkpoint as \[dataset\]\\Pretrained.pth

- Finetune

<code>python main.py --imbalance --no-cuda --dataset=cora --setting='newG_cls' --load=Pretrained.pth</code>


### Baselines
We provide five baselines in this code. They can be configured via the '--setting' arguments. Please refer to the 'get_parser()' function in utils.py.
- Oringinal model: Vanilla backbone models. '--setting='no''
- Over-sampling in raw input domain: Repeat nodes in the minority classes. '--setting='upsampling''
- Reweight: Give samples from minority classes a larger weight when calculating the loss. '--setting='reweight''
- SMOTE: Oversampling in raw input domain via SMOTE. '--setting='smote''.
- Embed-SMOTE: Perform SMOTE in the intermediate embedding domain. '--setting='embed_up''

Use over-sampling as an example: 

<code>python main.py --imbalance --no-cuda --dataset=cora --setting='upsampling'</code>

## Citation


If any problems occurs via running this code, please contact us at tkz5084@psu.edu.

Thank you!


