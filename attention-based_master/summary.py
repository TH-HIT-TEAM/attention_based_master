import tensorflow as tf

class Summary(object):
    def __init__(self, name):

        self.precision = tf.placeholder(tf.float32, name="precision")
        self.recall = tf.placeholder(tf.float32, name="recall")
        self.f1 = tf.placeholder(tf.float32, name="f1")

        with tf.name_scope(name):
            precision_summary = tf.summary.scalar("01_precision", self.precision)
            recall_summary = tf.summary.scalar("02_recall", self.recall)
            f1_summary = tf.summary.scalar("03_f1", self.f1)

        summary_list = [precision_summary, recall_summary, f1_summary]
        self.summary = tf.summary.merge(summary_list)
