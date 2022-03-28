import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    CNN网络结构设计
    一个embedding layer+一个convolution layer（Relu）+一个maxpooling层+softmax
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        # sequence_length 最长句子长度，不足长度就填充
        # num_classes：分类的类别数
        # vocab_size:字典词汇大小
        # embedding_size：词向量维度
        # filter_sizes：卷积核尺寸，简写为3,4,5。实际上为3*embedding_size
        # num_filters：每种尺寸的卷积核的个数
        # l2_reg_lambda=0.1 ：L2正则化参数

        # 定义占位符，先声明，后赋值
        # 句子矩阵，长为句子数量（自适应、样本个数），宽为sequence_length（句子固定宽度）
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")   # 输入数据 维度为batch_size*sequence_length.一段文本是一个样本
        # 存储对应分类结果 长度自适应、宽度为num_classes
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # L2正则化参数
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            # self.W为词向量词典，存储vocab_size个大小为embedding_size的词向量，随机初始化为-1到1之间的值
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W") # 随机初始化embedding矩阵
            # input_x是词的id序列
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)    # 查找一个张量里面 索引对应的元素
            # 第一个参数为一个张量或者索引、第二个参数为索引
            # embedded_chars是输入input_x对应的词向量表示，维度为[句子数量,sequence_length,embedding_size]
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            # tf.expand_dims(input,dim，name=None) 增加维度,主要是卷积2d输入是四维，这里是将词向量扩充一个维度，
            # 维度变为[句子数量, sequence_length, embedding_size, 1]，方便卷积

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []  # 五维向量，对于每种尺寸的卷积核，得到的结果是四维向量。
        for i, filter_size in enumerate(filter_sizes):  # 迭代索引及对应的尺寸
            with tf.name_scope("conv-maxpool-%s" % filter_size):   # 对于每种尺寸的卷积核，创建一个命名空间
                # 输入：batch_size（句子数） * sequence_length （句子定长）* embedding_size （对应宽度）*1（输入通道数）
                # 卷积尺寸 ：        filter_size（卷积核高度），embedding_size（卷积核宽度、词向量维度），1（图像通道数）， num_filters（输出通道数）
                # 卷积输出 ：batch_size * （sequence_length-filter_size+1） *  1 * num_filters
                # 池化尺寸 ：             （sequence_length-filter_size+1） *  1
                # 池化输出 ：batch_size * 1 * 1 * num_filters  三维矩阵

                # Convolution Layer
                # num_filters个（输出通道数）大小为filter_size*embedding_size的卷积核，输入通道数为1
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                # 卷积核宽、高、输入通道数、输出通道数—深度（卷积核的个数）
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,   # 输入、卷积参数、步长、填充方式-不填0
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID", # VALID窄卷积，SAME为等长卷积
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                # pooled 为 batch_size * 1 * num_filters。其中1个pool值为一个列表,如batch_size=3，num_filters = 2
                # pooled:[  [[1],[2]],  [[3],[4]],  [[5],[6]] ]  # 最里面的元素表示一个filter得到的一个特征值

                # [[3,4,5],[1,2,0]],每个元素为一个batch的num_filters个pool值
                pooled_outputs.append(pooled)    # 每个样本pooled_outputs中含有num_filters个数量的特征
                # pooled_outputs为五维矩阵[   ]
                #[len(filter_sizes), batch, height, width, channels = 1] # 对width，即词的角度串联

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 1)
        # 将不同核尺寸的对应特征进行拼接，如[[3]]与[[4]]拼接后就是[[34]]

        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        # 扁平化，压成一维，维度为batch_size * 卷积核总数

        # Add dropout
        with tf.name_scope("dropout"):    # dropout层，对池化后的结果做dropout
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")    # matmul(x, weights) + biases.
            self.predictions = tf.argmax(self.scores, 1, name="predictions")   # 返回索引

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            # 交叉熵
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            #losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
