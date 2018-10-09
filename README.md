# ChineseNRE
python 2.7
pytorch 0.4.0

中文实体关系抽取，对实体关系抽取不了解的可以先看<a href="https://blog.csdn.net/buppt/article/details/82961979">这篇文章</a>。顺便求star～

## 数据
中文实体关系抽取数据实在太难找了，data中是忘记在哪里找的人物关系数据集，一共11+1种关系。同学们如果有其他的数据集求分享～
```
unknown 0
父母 1
夫妻 2
师生 3
兄弟姐妹 4
合作 5
情侣 6
祖孙 7
好友 8
亲戚 9
同门 10
上下级 11
```

## 训练
模型使用的是lstm+attention模型。特征使用词向量+位置向量。

训练前先运行data文件夹中的 `data_util.py` 文件，将中文数据处理成pkl文件供模型使用。

然后运行`train.py`文件即可，可以在`train.py`文件中设置epoch、batch等参数，运行结束模型会储存到model文件夹中。

## 更新日志
2018-10-7 第一版，不定期进行修改与优化。
2018-10-9 添加准确率、召回率、f值的计算，将model从`train.py`中分离。
