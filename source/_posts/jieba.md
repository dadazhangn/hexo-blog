---
title: 优化jiba分词效率Demo
data: 2022-11-05
tags:
    - Jieba
    - 主题
    - demo
categories:
    - 教程
    - Jieba
top_img: 'linear-gradient(20deg, #0062be, #925696, #cc426e, #fb0347)'
cover: https://s2.loli.net/2024/02/14/vZyht9pO4kPVoKM.jpg

copyright_author: dadazhang
copyright_author_href: https://dadazhangn.com
copyright_url: https://dadazhangn.com
copyright_info: 此文章版权归公共所有，欢迎转载，


---

``` python
import time
import jieba

def split_words(content_list, stopword_set):
    cut_words_list = []
    start_time = time.time()

    for mail in content_list:
        # 使用 jieba.lcut_for_search 替代 jieba.lcut
        cut_words = jieba.lcut_for_search(mail)
        # 使用集合进行停用词检查
        cut_words = [word for word in cut_words if word not in stopword_set]
        cut_words_list.append(cut_words)

    print('jieba分词用时%.2f秒' % (time.time() - start_time))
    return cut_words_list
```


# 示例用法
``` python
    content_list = ["邮件1的内容", "邮件2的内容", "邮件3的内容"]
    stopword_list = ["停用词1", "停用词2", "停用词3"]
    stopword_set = set(stopword_list)

    result = split_words(content_list, stopword_set)
```