# CasFlow

![](https://img.shields.io/badge/TKDE-2021-blue)
![](https://img.shields.io/badge/python-3.9.7-green)
![](https://img.shields.io/badge/tensorflow-2.9.1-green)
![](https://img.shields.io/badge/cudatoolkit-11.2.2-green)
![](https://img.shields.io/badge/cudnn-8.1.0-green)
 
This repo provides a reference implementation of **CasFlow** as described in the paper:
> [CasFlow: Exploring Hierarchical Structures and Propagation Uncertainty for Cascade Prediction](https://doi.org/10.1109/TKDE.2021.3126475)  
> [Xovee Xu](https://www.xoveexu.com), [Fan Zhou](https://dblp.org/pid/63/3122-2.html), [Kunpeng Zhang](http://www.terpconnect.umd.edu/~kpzhang/), [Siyuan Liu](https://scholar.google.com/citations?user=Uhvt7OIAAAAJ&hl=en), and [Goce Trajcevski](https://dblp.org/pid/66/974.html)  
> IEEE Transactions on Knowledge and Data Engineering (TKDE), 2021

## Basic Usage

### Requirements

The code was tested with `python 3.9.7`, `tensorflow 2.9.1`, `cudatoolkit 11.2`, and `cudnn 8.1.0`. Install the dependencies via [Anaconda](https://www.anaconda.com/):

```shell
# create virtual environment
conda create --name casflow python=3.9 

# activate environment
conda activate casflow

# install tensorflow and other requirements
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
pip install -r requirements.txt
```

### Run the code
```shell
cd ./casflow

# generate information cascades
python gene_cas.py --input=./dataset/sample/

# generate cascade graph and global graph embeddings 
python gene_emb.py --input=./dataset/sample/

# run CasFlow model
python casflow.py --input=./dataset/sample/
```
More running options are described in the codes, e.g., 

- Using the Weibo dataset: `--input=./dataset/weibo/`
- Change observation time: `--observation_time=3600`

## Datasets

See some sample cascades in `./dataset/sample/`.

Datasets download link: [Google Drive](https://drive.google.com/file/d/1o4KAZs19fl4Qa5LUtdnmNy57gHa15AF-/view?usp=sharing) or [Baidu Drive (password: `1msd`)](https://pan.baidu.com/s/1tWcEefxoRHj002F0s9BCTQ).

The datasets we used in the paper are come from:

- [Twitter](http://carl.cs.indiana.edu/data/#virality2013) (Weng *et al.*, [Virality Prediction and Community Structure in Social Network](https://www.nature.com/articles/srep02522), Scientific Report, 2013).
- [Weibo](https://github.com/CaoQi92/DeepHawkes) (Cao *et al.*, [DeepHawkes: Bridging the Gap between 
Prediction and Understanding of Information Cascades](https://dl.acm.org/doi/10.1145/3132847.3132973), CIKM, 2017). You can also download Weibo dataset [here](https://drive.google.com/file/d/1fgkLeFRYQDQOKPujsmn61sGbJt6PaERF/view?usp=sharing) in Google Drive.  
- [APS](https://journals.aps.org/datasets) (Released by *American Physical Society*, obtained at Jan 17, 2019).  

## Cite

If you find **CasFlow** useful for your research, please consider citing us 😘:

    @article{xu2021casflow,  
      author = {Xovee Xu and Fan Zhou and Kunpeng Zhang and Siyuan Liu and Goce Trajcevski},  
      title = {Cas{F}low: Exploring Hierarchical Structures and Propagation Uncertainty for Cascade Prediction},
      journal = {IEEE Transactions on Knowledge and Data Engineering (TKDE)},
      year = {2021}, 
      volume = {35}, 
      number = {4}, 
      pages={3484-3499}, 
      doi = {10.1109/TKDE.2021.3126475}, 
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
    

We also have a [survey paper](https://dl.acm.org/doi/10.1145/3433000) you might be interested:


    @article{zhou2021survey,
      author = {Fan Zhou and Xovee Xu and Goce Trajcevski and Kunpeng Zhang}, 
      title = {A Survey of Information Cascade Analysis: Models, Predictions, and Recent Advances}, 
      journal = {ACM Computing Surveys}, 
      volume = {54},
      number = {2},
      year = {2021},
      articleno = {27},
      numpages = {36},
      doi = {10.1145/3433000},
    }

## Contact

For any questions please open an issue or drop an email to: `xovee at live.com`
