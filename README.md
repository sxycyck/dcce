# DCCE

This code is implementation of our paper **A Robust Dual-debiasing VQA Model based on Counterfactual Causal Effect**. This code is implemented on the basis of [RUBi](https://github.com/cdancette/rubi.bootstrap.pytorch).

## 

## ![model](D:\DIRS\深度因果推理学习\dcce-master\model.png)

## Install python environment

We implement expereiments based on [block.bootstrap.pytorch](https://github.com/Cadene/block.bootstrap.pytorch). The installation details are as follows:

```
conda create --name dcce python=3.7
conda activate dcce
git clone --recursive https://github.com/sxycyck/dcce.git
cd dcce-master
pip install -r requirements.txt
```

### Notes

if  `no module named 'block.external' ` occurs, you can copy folder [external](external) to path `site-packages/block/external`.


## Download datasets

```
bash dcce/datasets/scripts/download_vqa2.sh
bash dcce/datasets/scripts/download_vqacp2.sh
```

### Dataset directory structure

```
data/vqa
	├── vqa2/raw/annotations
	├── vqacp2/raw/annotations
	├── coco/extract_rcnn/2018-04-27_bottom-up-attention_fixed_36
		├──COCO_test2015_000000000014.jpg.pth
		├──COCO_test2015_000000000016.jpg.pth
		.....
		
```

## Quick start

### Train a model

You can train a model by using following commands in directory `dcce-master`

```
python -m bootstrap.run -o  $CONFIG_FILE
```

The CONFIG_FILE can be loaded options in YAML format. For example, you can train our best model on VQACP v2 by running:

```
python -m bootstrap.run -o options/vqacp2/updn_s_TIE_d_TIE_wd.yaml
```

After training, the results are saved in `logs/vqacp2/updn\_s\_TIE\_d\_TIE\_wd`.

You also can create a new yaml file to start training procedure. Many options about our experiments are available in [options](options). The language bias include shortcut bias and distribution bias. The way of mitigating bias include TIE and NIE. For instance, if using TIE for mitigating shortcut bias, and NIE for mitigating distribution bias, the configuration file will be [**updn_s_TIE_d_NIE_wd.yaml**](options/vqacp2/updn_s_TIE_d_NIE_wd.yaml) .

File `_dcce_val_oe.json` records the accuracy metrics in VQACP v2 validation set by using our methods.

### Evaluate a model

For a model trained on VQACP v2, you can evaluate trained model on the validation set using commands:

```
python -m bootstrap.run  \\

-o options/vqacp2/updn_s_TIE_d_TIE_wd.yaml  \\

--exp.resume last \\\

--dataset.train_split  ' '  \\\

--dataset.eval_split val \\\

--misc.logs_name test
```

The test accuracy metrics is written in `test_dcce_val_oe.json`.

### Resume training

Once the training procedure is interrupted, you can resume by running:

```
python -m bootstrap.run -o options/vqacp2/updn_s_TIE_d_TIE_wd.yaml  --exp.resume last
```

### Use a specific GPU

If  you need to use a specific GPU to train model, you need to

```bash
CUDA_VISIBLE_DEVICES=0 python -m bootstrap.run -o options/vqacp2/updn_s_TIE_d_TIE_wd.yaml
```

## Acknowledgment

Special thanks to the authors of [CFVQA](https://github.com/yuleiniu/cfvqa), [RUBi](https://github.com/cdancette/rubi.bootstrap.pytorch), [BLOCK](https://github.com/Cadene/block.bootstrap.pytorch), and [bootstrap.pytorch](https://github.com/Cadene/bootstrap.pytorch), and the datasets used in this research project.







