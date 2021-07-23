# Input

A txt file, each line is a cascade, following is a cascade example:

    24800	1849561	1464745294	17	1849561:0 1849561/1849561/1849562:36100 1849561/1849561/420444:31422 1849561/1849561/601421:29156 1849561/1849561/1445451:26631 1849561/1849561:26332 1849561/1445451:18869 1849561/125921/545519/619369:18307 1849561/538011:14565 1849561/1849563:4295 1849561/125921/467482:3655 1849561/85207:2501 1849561/125921/545519:7698 1849561/601595:1739 1849561/1849564:900 1849561/125921:350 1849561/1299382:360

There's five parts each line, splited by `\t`: 
1. the cascade id
2. the original user id
3. publication timestamp
4. number of participants (e.g., retweets)
5. participants, splited by space ` `, take `1839561/125921/467482:3655` as an example, it means user `467482` retweet user `1839561`'s tweet by through user `125921`'s retweet, after 3655 seconds since the publication of the tweet. 

> Note: `xovee` is just a sample dataset.