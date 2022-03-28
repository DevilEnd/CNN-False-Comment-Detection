# 网络训练

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn

os.environ["CUDA_VISIBLE_DEVICES"] = "0" #‘1’为指定GPU，‘0’为CPU
# 命令行参数
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", 0.1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("data_file", "fake reviews dataset.csv", "Data source for the train data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding")
tf.flags.DEFINE_string("filter_sizes", "3", "Comma-separated filter sizes")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability")
tf.flags.DEFINE_float("l2_reg_lambda", 0.1, "L2 regularization lambda")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size")
tf.flags.DEFINE_integer("num_epochs", 40, "Number of training epochs")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()
# print("\nParameters:")
# for attr, value in sorted(FLAGS.__flags.items()):
#     print("{}={}".format(attr.upper(), value))
# print("")

# 数据预处理，包括：构建词典，将文本变成词id序列，随机打乱数据、划分数据集（训练集、验证集、可用交叉验证)
def preprocess():
    # Data Preparation
    # ==================================================

    # Load data
    print("Loading data...")
    x_text, y = data_helpers.load_data_and_labels(FLAGS.data_file)
    # x_text 为训练集，以列表的形式存储，每一个元素为字符串，处理过的文本
    # y为每个文本对应的标签，以列表存储，每个元素为一个列表（one_hot向量表示）

    # Build vocabulary 词典的id是从1开始的，0是留给那些未出现的词以及作填充长度用的
    max_document_length = max([len(x.split(" ")) for x in x_text])    # 统计每个文本的长度，并获得文本最大的长度
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    # 构建词典，文本中每个词分配个id
    x = np.array(list(vocab_processor.fit_transform(x_text)))
    # 将文本表示成词向量的形式

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = np.array(y)[shuffle_indices]

    # Split train/test set
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    del x, y, x_shuffled, y_shuffled

    # 获取词典的大小以及训练集、验证集的比例
    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, y_train, vocab_processor, x_dev, y_dev

def train(x_train, y_train, vocab_processor, x_dev, y_dev):
    # Training
    # ==================================================

    with tf.Graph().as_default():
        # tf.ConfigProto用在创建session的时候，用来对session进行参数配置
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,   # 记录设备指派情况
          log_device_placement=FLAGS.log_device_placement)   # 自动选择运行设备
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            # 初始化为0，每次自动加1

            # 使用噪声线性余弦函数令学习率衰减
            #learing_rate1 = tf.train.noisy_linear_cosine_decay(
            #learning_rate=0.001, global_step=global_step, decay_steps=00)

            # Adam分类器
            optimizer = tf.train.AdamOptimizer(1e-3)
            # 计算梯度，返回元组列表[ (gradient,variable)..]
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            # 根据梯度更新变量
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            #train_op = optimizer.minimize(cnn.loss，global_step=global_step) 上面两个步骤等价于这一步，分开方便对梯度进行约束

            # Keep track of gradient values and sparsity (optional) 可视化
            # 查看一个张量在训练过程中值的分布情况时，可通过tf.summary.histogram()将其分布情况以直方图的形式在TensorBoard直方图仪表板上显示
            grad_summaries = []
            for g, v in grads_and_vars:     # [ (gradient,variable)..] g:梯度，v:变量
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # tf.summary()的各类方法，能够保存训练过程以及参数分布图并在tensorboard显示

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # 定义了一个函数，输入为1个batch
            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                # 梯度更新（更新模型），步骤加一，存储数据，计算一个batch的损失，计算一个batch的准确率
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                # step总次数的计算过程为 每一轮含有的batch数目（文本总数量/一个batch含有的文本的数量）*总轮数（200）
                train_summary_writer.add_summary(summaries, step)

            # 定义了一个函数，用于验证集，输入为一个batch
            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches（生成器），得到一个generator，每一次返回一个batch，没有构成list[batch1,batch2,batch3,...]
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # 控制迭代次数
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

def main(argv=None):
    x_train, y_train, vocab_processor, x_dev, y_dev = preprocess()
    train(x_train, y_train, vocab_processor, x_dev, y_dev)

if __name__ == '__main__':
    tf.app.run()