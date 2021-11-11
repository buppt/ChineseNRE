# coding=utf-8
import codecs
import re
import pandas as pd
import numpy as np
from collections.abc import Iterable
from collections import deque

relation2id = {
    "Other": 0,
    "Cause-Effect": 1,
    "Instrument-Agency": 2,
    "Product-Producer": 3,
    "Content-Container": 4,
    "Entity-Origin": 5,
    "Entity-Destination": 6,
    "Component-Whole": 7,
    "Member-Collection": 8,
    "Message-Topic": 9
}
datas = deque()
labels = deque()
entity1 = deque()
entity2 = deque()
with codecs.open('TRAIN_FILE.TXT', 'r', 'utf-8') as tra:
    linenum = 0
    for line in tra:
        linenum += 1
        if linenum % 4 == 1:
            line = line.split('\t')[1]

            word_arr = line[1:-4].split()
            for index in range(len(word_arr)):
                if "<e1>" in word_arr[index]:
                    entity1.append(index)
                elif "<e2>" in word_arr[index]:
                    entity2.append(index)

            line = line.replace("<e1>", "")
            line = line.replace("</e1>", "")
            line = line.replace("<e2>", "")
            line = line.replace("</e2>", "")
            line = re.sub(r'[^\w\s]', '', line)

            datas.append(line[1:-2].split())
        elif linenum % 4 == 2:
            if line == "Other\r\n":
                labels.append(0)
            else:
                line = line.split('(')
                labels.append(relation2id[line[0]])

        else:
            continue

print(len(datas), len(labels), len(entity1), len(entity2))
print(datas[0], labels[0], entity1[0], entity2[0])


def flatten(x):
    result = []
    for el in x:
        if isinstance(x, Iterable) and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result


all_words = flatten(datas)
sr_allwords = pd.Series(all_words)
sr_allwords = sr_allwords.value_counts()
#print sr_allwords
set_words = sr_allwords.index
set_ids = range(1, len(set_words) + 1)
word2id = pd.Series(set_ids, index=set_words)
id2word = pd.Series(set_words, index=set_ids)

word2id["BLANK"] = len(word2id) + 1
word2id["UNKNOW"] = len(word2id) + 1
print(word2id)
max_len = 70
senssslen = 0


def X_padding(words):
    ids = []
    for word in words:
        if word in word2id:
            ids.append(word2id[word])
        else:
            ids.append(word2id["UNKNOW"])

    if len(ids) >= max_len:
        return ids[:max_len]
    ids.extend([word2id["BLANK"]] * (max_len - len(ids)))
    return ids


def pos_padding(index):
    ids = []
    for i in range(max_len):
        ids.append(i - index + max_len)
    if max_len - index < 0:
        print(index, ids)
    return ids


x = deque()
pos_e1 = deque()
pos_e2 = deque()
for index in range(len(datas)):
    x.append(X_padding(datas[index]))
    pos_e1.append(pos_padding(entity1[index]))
    pos_e2.append(pos_padding(entity2[index]))

x = np.asarray(x)
y = np.asarray(labels)
pos_e1 = np.asarray(pos_e1)
pos_e2 = np.asarray(pos_e2)

print(x.shape, y.shape, pos_e1.shape, pos_e2.shape)

import pickle
with open('../engdata_train.pkl', 'wb') as outp:
    pickle.dump(word2id, outp)
    pickle.dump(id2word, outp)
    pickle.dump(relation2id, outp)
    pickle.dump(x, outp)
    pickle.dump(y, outp)
    pickle.dump(pos_e1, outp)
    pickle.dump(pos_e2, outp)
print('** Finished saving train data.')

datas = deque()
labels = deque()
entity1 = deque()
entity2 = deque()
with codecs.open('TEST_FILE_FULL.TXT', 'r', 'utf-8') as tra:
    linenum = 0
    for line in tra:
        linenum += 1
        if linenum % 4 == 1:
            line = line.split('\t')[1]

            word_arr = line[1:-4].split()
            for index in range(len(word_arr)):
                if "<e1>" in word_arr[index]:
                    entity1.append(index)
                elif "<e2>" in word_arr[index]:
                    entity2.append(index)

            line = line.replace("<e1>", "")
            line = line.replace("</e1>", "")
            line = line.replace("<e2>", "")
            line = line.replace("</e2>", "")
            line = re.sub(r'[^\w\s]', '', line)

            datas.append(line[1:-2].split())
        elif linenum % 4 == 2:
            if line == "Other\r\n":
                labels.append(0)
            else:
                line = line.split('(')
                labels.append(relation2id[line[0]])

        else:
            continue

x = deque()
pos_e1 = deque()
pos_e2 = deque()
for index in range(len(datas)):
    x.append(X_padding(datas[index]))
    pos_e1.append(pos_padding(entity1[index]))
    pos_e2.append(pos_padding(entity2[index]))

x = np.asarray(x)
y = np.asarray(labels)
pos_e1 = np.asarray(pos_e1)
pos_e2 = np.asarray(pos_e2)

print(x.shape, y.shape, pos_e1.shape, pos_e2.shape)

import pickle
with open('../engdata_test.pkl', 'wb') as outp:
    pickle.dump(x, outp)
    pickle.dump(y, outp)
    pickle.dump(pos_e1, outp)
    pickle.dump(pos_e2, outp)
print('** Finished saving train data.')
