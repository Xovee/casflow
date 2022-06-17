# Dataset 

## Description

A txt file, each line is an information cascade, following is a cascade example:

    24800	1849561	1464745294	17	1849561:0 1849561/1849561/1849562:36100 1849561/1849561/420444:31422 1849561/1849561/601421:29156 1849561/1849561/1445451:26631 1849561/1849561:26332 1849561/1445451:18869 1849561/125921/545519/619369:18307 1849561/538011:14565 1849561/1849563:4295 1849561/125921/467482:3655 1849561/85207:2501 1849561/125921/545519:7698 1849561/601595:1739 1849561/1849564:900 1849561/125921:350 1849561/1299382:360

There's five parts each line, splited by `\t`: 
1. the cascade id
2. the original user id
3. publication timestamp
4. number of participants (e.g., retweets)
5. participants, splited by space ` `, take `1839561/125921/467482:3655` as an example, it means user `467482` retweet user `1839561`'s tweet by through user `125921`'s retweet, after 3655 seconds since the publication of the tweet. 

## Observation Time

| Dataset | Setting 1 | # in Code | Setting 2 | # in Code |
|:--------|:----------|:----------|:----------|:----------|
| Twtter  | 1 day     | 86400     | 2 days    | 172800    |
| Weibo   | 0.5 hour  | 1800      | 1 hour    | 3600      |
| APS     | 3 years   | 1095      | 5 years   | 1826      |

## Prediction Time

| Dataset | Time     | # in Code |
|:--------|:---------|:----------|
| Twitter | 32 days  | 2764800   |
| Weibo   | 24 hours | 86400     |
| APS     | 20 years | 7305      |

## Dataset Split Validation

For reproducing our results, make sure the dataset split results are correct (consistent with the table below). Followings are cascade IDs in the first few lines of each dataset. 

> For historical reasons, baselines such as DeepHawkes and CasCN use 8 AM to 7 PM settings (which should be 8 AM to 6 PM) and they also used `time.localtime()` function to process UTC timestamps (might be UTC+8). If you want to compare CasFlow to other baselines, make sure their datasets (train, val, and test) are identical for training and testing. 

| Dataset | Obvervation Time | Training Set                                 | Validation Set                             | Test Set                                   |
|:--------|:-----------------|:---------------------------------------------|:-------------------------------------------|:-------------------------------------------|
| Twitter | 1 day            | 7355 ...<br>44241 ...<br>401 ...<br>...      | 26 ...<br>43 ...<br>44 ...<br>...          | 23 ...<br>33 ...<br>39 ...<br>...          |
| Twitter | 2 days           | 1218 ...<br>79377 ...<br>78968 ...<br>...    | 40 ...<br>46 ...<br>77 ...<br>...          | 7 ...<br>33 ...<br>34 ...<br>...           |
| Weibo   | 0.5 hour         | 41476 ...<br>19109 ...<br>42289 ...<br>...   | 14709 ...<br>14731 ...<br>14743 ...<br>... | 14713 ...<br>14720 ...<br>14724 ...<br>... |
| Weibo   | 1 hour           | 20674 ...<br>69746 ...<br>31805 ...<br>...   | 14712 ...<br>14722 ...<br>14726 ...<br>... | 14719 ...<br>14724 ...<br>14731 ...<br>... |
| APS     | 3 years          | 103460 ...<br>34784 ...<br>154897 ...<br>... | 2742 ...<br>3418 ...<br>3659 ...<br>...    | 3039 ...<br>3176 ...<br>3302 ...<br>...    |
| APS     | 5 years          | 170722 ...<br>223749 ...<br>60436 ...<br>... | 2693 ...<br>2959 ...<br>3011 ...<br>...    | 2754 ...<br>3001 ...<br>3098 ...<br>...    |

## Running Scripts

```shell
# twitter 1 day
python gene_cas.py --input=./dataset/twitter/ --observation_time=86400 --prediction_time=2764800
python gene_emb.py --input=./dataset/twitter/ --observation_time=86400
python casflow.py --input=./dataset/twitter/
# twitter 2 days 
python gene_cas.py --input=./dataset/twitter/ --observation_time=172800 --prediction_time=2764800
python gene_emb.py --input=./dataset/twitter/ --observation_time=172800
python casflow.py --input=./dataset/twitter/

# weibo 0.5 hour
python gene_cas.py --input=./dataset/weibo/ --observation_time=1800 --prediction_time=86400
python gene_emb.py --input=./dataset/weibo/ --observation_time=1800
python casflow.py --input=./dataset/weibo/
# weibo 1 hour
python gene_cas.py --input=./dataset/weibo/ --observation_time=3600 --prediction_time=86400
python gene_emb.py --input=./dataset/weibo/ --observation_time=3600
python casflow.py --input=./dataset/weibo/

# aps 3 years
python gene_cas.py --input=./dataset/aps/ --observation_time=1095 --prediction_time=7305
python gene_emb.py --input=./dataset/aps/ --observation_time=1095
python casflow.py --input=./dataset/aps/
# aps 5 years
python gene_cas.py --input=./dataset/aps/ --observation_time=1826 --prediction_time=7305
python gene_emb.py --input=./dataset/aps/ --observation_time=1826
python casflow.py --input=./dataset/aps/
```