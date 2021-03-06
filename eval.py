#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv

tp = 0.0  # 记录预测为虚假评论且实际为虚假评论的结果数
fp = 0  # 记录预测为虚假评论但实际为真实评论的结果数
tn = 0.0  # 记录预测为真实评论且实际为真实评论的结果数
fn = 0  # 记录预测为真实评论但实际为虚假评论的结果数
# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("data_file", "fake reviews dataset.csv", "Data source for the train data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", r"E:\cnn-text-classification-tf-master\cnn-text-classification-tf-master\runs\1648175989\checkpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# CHANGE THIS: Load data. Load your own data here
if FLAGS.eval_train:
    x_raw, y_test = data_helpers.load_data_and_labels(FLAGS.data_file)
    y_test = np.argmax(y_test, axis=1)
else:
    x_raw = ["a masterpiece four years in the making", "everything is off."]
    y_test = [1, 0]

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy if y_test is defined
#if y_test is not None:
    #correct_predictions = float(sum(all_predictions == y_test))
    #print("Total number of test examples: {}".format(len(y_test)))
    #print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

for j in range(len(y_test)):
        # 验证时出现四种情况分别对应四个变量存储
        if all_predictions[j] == y_test[j] == 1:
            tp += 1
        elif all_predictions[j] == y_test[j] == 0:
            tn += 1
        elif all_predictions[j] == 1 and y_test[j] == 0:
            fp += 1
        else:
            fn += 1

accuracy = tp / (tp + fp)  # 准确率
recall = tp / (tp + fn)  # 召回率
print("All Accuracy: {:g}".format((tp + tn)/float(len(y_test))))
print('accuracy: {:g}'.format(accuracy))
print('recall: {:g}'.format(recall))
print('F1 score: {:g}'.format(2 * accuracy * recall / (accuracy + recall)))  # 计算F1值