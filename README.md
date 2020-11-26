# CasFlow

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19zQrAIo-yyfkH8m95UmdepzSryxHHs_a?usp=sharing)
![](https://img.shields.io/badge/python-3.7-green)
![](https://img.shields.io/badge/tensorflow-2.0.0a-green)
![](https://img.shields.io/badge/cudatoolkit-10.0-green)
![](https://img.shields.io/badge/cudnn-7.6.5-green)

This repo provides a reference implementation of **CasFlow** as described in the paper:
> CasFlow: Exploring Hierarchical Structures and Propagation Uncertainty for Cascade Prediction  
> [Fan Zhou](https://dblp.org/pid/63/3122-2.html), [Xovee Xu](https://xovee.cn), [Kunpeng Zhang](http://www.terpconnect.umd.edu/~kpzhang/), [Siyuan Liu](https://scholar.google.com/citations?user=Uhvt7OIAAAAJ&hl=en) and [Goce Trajcevski](https://dblp.org/pid/66/974.html)  
> Submitted for review  

## Basic Usage

### Requirements

The code was tested with `python 3.7`, `tensorflow-gpu 2.0.0a`, `cudatookkit 10.0`, and `cudnn 7.6.5`. Install the dependencies via [Anaconda](https://www.anaconda.com/):

```shell
# create virtual environment
conda create --name casflow python=3.7 cudatoolkit=10.0 cudnn=7.6.5

# activate environment
conda activate casflow

# install other requirements
pip install -r requirements.txt
```

### Run the code
```shell
cd casflow

# generate information cascades
python gene_cas.py

# generate cascade and global graph embeddings 
python gene_emb.py

# run the CasFlow model
python casflow.py
```
More running options are described in the codes, e.g., `--input=./dataset/weibo/`

### Run code in Google Colab

You can also run the code in [Google Colab](https://colab.research.google.com/drive/19zQrAIo-yyfkH8m95UmdepzSryxHHs_a?usp=sharing). 

#### Datasets

See some sample cascades in `./dataset/xovee/`.

The datasets we used in the paper can be obtained here:

- [Twitter](http://carl.cs.indiana.edu/data/#virality2013) (Weng *et al.*, [Virality Prediction and Community Structure in Social Network](https://www.nature.com/articles/srep02522), Scientific Report, 2013).
- [Weibo](https://github.com/CaoQi92/DeepHawkes) (Cao *et al.*, [DeepHawkes: Bridging the Gap between 
Prediction and Understanding of Information Cascades](https://dl.acm.org/doi/10.1145/3132847.3132973)., CIKM, 2017). You can also download Weibo dataset [here](https://drive.google.com/file/d/1fgkLeFRYQDQOKPujsmn61sGbJt6PaERF/view?usp=sharing) in Google Drive.  
- [APS](https://journals.aps.org/datasets) (Released by *American Physical Society*, obtained at Jan 17, 2019). 

## Cite

If you find **CasFlow** useful for your research, please consider citing us 😘:

    @article{zhou2020casflow,  
      author = {Fan Zhou and Xovee Xu and Kunpeng Zhang and Siyuan Liu and Goce Trajcevski},  
      title = {CasFlow: Exploring Hierarchical Structures and Propagation Uncertainty for Cascade Prediction},
      booktitle = {Submitted for publication},
      year = {2020}, 
    }
    
This paper is an extension of [VaCas](https://doi.org/10.1109/INFOCOM41043.2020.9155349):

    @inproceedings{zhou2020variational,
      author = {Fan Zhou and Xovee Xu and Kunpeng Zhang and Goce Trajcevski and Ting Zhong},
      title = {Variational Information Diffusion for Probabilistic Cascades Prediction}, 
      booktitle = {IEEE International Conference on Computer Communications (INFOCOM)},
      year = {2020},
      pages = {1618--1627},
      doi = {10.1109/INFOCOM41043.2020.9155359},
    }
    
We also have a [survey paper](https://arxiv.org/abs/2005.11041) you might be interested:

    @article{zhou2020survey,
      author = {Fan Zhou and Xovee Xu and Goce Trajcevski and Kunpeng Zhang}, 
      title = {A Survey of Information Cascade Analysis: Models, Predictions and Recent Advances}, 
      journal = {ACM Computing Surveys (CSUR)}, 
      year = {2020},
    }
