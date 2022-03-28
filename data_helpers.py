import numpy as np
import re
import os
import csv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def clean_str(string):
    """
    数据预处理
    """
    stop_words = set(stopwords.words('english'))

    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    # 去除停用词
    word_tokens = word_tokenize(string)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    string=' '.join(filtered_sentence)

    return string.strip().lower()


def load_data_and_labels(data_file):
    """
    读取标签及评论数据
    """
    x_text = []
    y = []

    with open(data_file, 'r', encoding='utf-8') as csvfile:
        read = csv.reader(csvfile)
        # 按行读取CSV文件
        for index,info in enumerate(read):
            if index!=0:   #判断，跳过表头
                if info[0] == 'Books_5':
                    x_text.append(info[3])
                    if info[2] == 'CG':
                        y.append([0, 1])  # 假，01,onehot编码
                    else:
                        y.append([1, 0])  # 真，10

    x_text = [clean_str(sent) for sent in x_text]

    return [x_text, y]

#  batches = data_helpers.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size=128, FLAGS.num_epochs=100)
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    定义一个函数，输出batch样本，参数为data（包括feature和label），batchsize，epoch
    """
    data = np.array(data)
    # data是个列表，里面的元素的数据类型是一致的，用数组np.array存储
    # type(data)返回data的数据类型，data.dtype返回数组中内容的数据类型。
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1  # 每次迭代训练所有数据，分成多少个batch
    for epoch in range(num_epochs):     # 在每一轮迭代过程中，都打乱数据
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))    # 打乱索引
            shuffled_data = data[shuffle_indices]    # 数据打乱
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):    # 对于每个batch的数据，获得batch内的起始与终止的位置
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
            # yield，在for循环执行时，每次返回一个batch的data，占用的内存为常数
