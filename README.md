# CasFlow

This repo provides a reference implementation of **CasFlow** as described in the paper:
> Exploring Hierarchical Structures and Propagation Uncertainty for Cascade Prediction.  
> Fan Zhou, Xovee Xu, Kunpeng Zhang and Goce Trajcevski.  
> Submitted for publication.

## Basic Usage

### Requirements

The code was tested with Python 3.7, `tensorflow-gpu` 2.0.0a and Cuda 10.0. Install the dependencies:

```python
pip install -r requirements.txt
```
d
### Run the code
```shell
> cd casflow
> python gene_cascade.py
> python preprocessing.py
> python casflow.py
```

### Run code in Google Colab

You can also run the code in [Google Golab](https://colab.research.google.com/drive/19zQrAIo-yyfkH8m95UmdepzSryxHHs_a?usp=sharing). 

#### Options
You may change the model settings manually in `config.py` or directly into the codes. 

#### Datasets

See some sample cascades in `./dataset/xovee/`.

The datasets we used in the paper can be obtained here:

- [Twitter](http://carl.cs.indiana.edu/data/#virality2013) (Weng *et al.*, [Virality Prediction and Community Structure in Social Network](https://www.nature.com/articles/srep02522), Scientific Report, 2013).
- [Weibo](https://github.com/CaoQi92/DeepHawkes) (Cao *et al.*, [DeepHawkes: Bridging the Gap between 
Prediction and Understanding of Information Cascades](https://dl.acm.org/doi/10.1145/3132847.3132973)., CIKM, 2017). You can also download Weibo dataset [here](https://drive.google.com/file/d/1fgkLeFRYQDQOKPujsmn61sGbJt6PaERF/view?usp=sharing) in Google Drive.  
- [APS](https://journals.aps.org/datasets) (Released by *American Physical Society*, obtained at Jan 17, 2019). 

## Cite

If you find **CasFlow** useful for your research, please consider citing us 😘:

    @inproceedings{zhou2020exploring,  
      author = {Zhou, Fan and Xu, Xovee and Zhang, Kunpeng and Trajcevski, Goce},  
      title = {Exploring Hierarchical Structures and Propagation Uncertainty for Cascade Prediction},
      booktitle = {Submitted for publication},
      year = {2020}
    }
