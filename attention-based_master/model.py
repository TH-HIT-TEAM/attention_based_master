import tensorflow as tf
import numpy as np
import function as fun
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes


class Model(object):
    def __init__(self, config):
        max_len_list = config.max_len_list
        max_len_x = max_len_list[0]
        # max_len_y = max_len_list[1]
        # max_len_z = max_len_list[2]
        # max_len_p = max_len_list[3]
        class_num = config.class_num
        embedding_dim = config.embedding_dim
        pos_embedding_dim = config.pos_embedding_dim
        distance_embedding_dim = config.distance_embedding_dim
        hidden_neural_size = config.hidden_neural_size
        attention_size = config.attention_size
        vocab_size_x = config.vocab_size_x
        vocab_size_y = config.vocab_size_y
        vocab_size_z = config.vocab_size_z
        vocab_size_yz = config.vocab_size_yz
        vocab_size_p = config.vocab_size_p


        self.input_x = tf.placeholder(tf.int32, [None, max_len_x], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [None, max_len_x], name="input_y")
        self.input_z = tf.placeholder(tf.int32, [None, max_len_x], name="input_z")
        self.input_p = tf.placeholder(tf.int32, [None, max_len_x], name="input_p")
        self.input_mask_x = tf.placeholder(tf.float32, [None, max_len_x], name="input_mask_x")
        self.input_mask_y = tf.placeholder(tf.float32, [None, max_len_x], name="input_mask_y")
        self.input_mask_z = tf.placeholder(tf.float32, [None, max_len_x], name="input_mask_z")
        self.input_mask_p = tf.placeholder(tf.float32, [None, max_len_x], name="input_mask_p")
        self.input_label = tf.placeholder(tf.float32, [None, class_num], name="input_label")
        self.input_label_bi = tf.placeholder(tf.float32, [None, 2], name="input_label_bi")
        self.length_x = tf.placeholder(tf.int64, [None], name="length_x")
        self.length_y = tf.placeholder(tf.int64, [None], name="length_y")
        self.length_z = tf.placeholder(tf.int64, [None], name="length_z")
        self.length_yz = self.length_y
        self.length_p = tf.placeholder(tf.int64, [None], name="length_p")

        self.cur_num_epoch = tf.Variable(0, name="cur_num_epoch", trainable=False)
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.lr = tf.Variable(0.0, name="lr", trainable=False)


        # embedding layer ----------------------------------------------------------------------------------------------
        with tf.device("/cpu:0"), tf.name_scope("embedding_layer"):
            # self.embedding_x = tf.placeholder(tf.float32, [vocab_size_x, embedding_dim], name="embedding_x")
            # self.embedding_y = tf.placeholder(tf.float32, [vocab_size_y, distance_embedding_dim], name="embedding_y")
            # self.embedding_z = tf.placeholder(tf.float32, [vocab_size_z, distance_embedding_dim], name="embedding_z")
            # self.embedding_yz = tf.placeholder(tf.float32, [vocab_size_yz, distance_embedding_dim], name="embedding_yz")
            self.embedding_x = tf.Variable(tf.random_uniform([vocab_size_x, embedding_dim], -1.0, +1.0), name="embedding_x")
            self.embedding_y = tf.Variable(tf.random_uniform([vocab_size_y, distance_embedding_dim], -1.0, +1.0), name="embedding_y")
            self.embedding_z = tf.Variable(tf.random_uniform([vocab_size_z, distance_embedding_dim], -1.0, +1.0), name="embedding_z")
            self.embedding_yz = tf.Variable(tf.random_uniform([vocab_size_yz, distance_embedding_dim], -1.0, +1.0), name="embedding_yz")
            self.embedding_p = tf.Variable(tf.random_uniform([vocab_size_p, pos_embedding_dim], -1.0, +1.0), name="embedding_p")
            self.embedding = [self.embedding_x, self.embedding_yz, self.embedding_p]
            input_x = tf.nn.embedding_lookup(self.embedding_x, self.input_x)
            input_y = tf.nn.embedding_lookup(self.embedding_yz, self.input_y)
            input_z = tf.nn.embedding_lookup(self.embedding_yz, self.input_z)
            input_p = tf.nn.embedding_lookup(self.embedding_p, self.input_p)
            # input_xyz = tf.concat([input_x, input_y, input_z], 2)
            input_yz = tf.concat([input_y, input_z], 2)

            # print ("input_x shape: ", input_x.get_shape())
            # print ("input_y shape: ", input_y.get_shape())
            # print ("input_z shape: ", input_z.get_shape())
            # # print ("input_xyz shape: ", input_xyz.get_shape())
            # print ("input_yz shape: ", input_yz.get_shape())
            # print ("input_p shape: ", input_p.get_shape())


# ===================================== x ==============================================================================

        # LSTM layer x 1 -----------------------------------------------------------------------------------------------
        with tf.variable_scope('lstm_layer_x_1'):
            # cell_f_x_1 = tf.nn.rnn_cell.LSTMCell(num_units=hidden_neural_size, state_is_tuple=True)
            # cell_b_x_1 = tf.nn.rnn_cell.LSTMCell(num_units=hidden_neural_size, state_is_tuple=True)
            # cell_f_x = tf.nn.rnn_cell.GRUCell(num_units=hidden_neural_size)
            # cell_b_x = tf.nn.rnn_cell.GRUCell(num_units=hidden_neural_size)
            cell_f_x_1 = tf.contrib.rnn.core_rnn_cell.LSTMCell(num_units=hidden_neural_size, state_is_tuple=True)
            cell_b_x_1 = tf.contrib.rnn.core_rnn_cell.LSTMCell(num_units=hidden_neural_size, state_is_tuple=True)
            outputs_x_1, states_x_1 = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_f_x_1,
                cell_bw=cell_b_x_1,
                dtype=tf.float32,
                sequence_length=self.length_x,
                inputs=input_x
            )
            output_fw_x_1, output_bw_x_1 = outputs_x_1
            # states_fw_x_1, states_bw_x_1 = states_x_1
            out_put_x_1 = (output_fw_x_1 + output_bw_x_1)
            # out_put_x = tf.concat(2, [output_fw_x, output_bw_x])
            out_put_x_1 = tf.multiply(out_put_x_1, self.input_mask_x[:, :, None])
            out_put_x = out_put_x_1

        # # LSTM layer x 2 -------------------------------------------------------------------------------------------------
        # with tf.variable_scope('lstm_layer_x_2'):
        #     cell_f_2 = tf.nn.rnn_cell.LSTMCell(num_units=hidden_neural_size, state_is_tuple=True)
        #     cell_b_2 = tf.nn.rnn_cell.LSTMCell(num_units=hidden_neural_size, state_is_tuple=True)
        #     outputs_x_2, states_x_2 = tf.nn.bidirectional_dynamic_rnn(
        #         cell_fw=cell_f_2,
        #         cell_bw=cell_b_2,
        #         dtype=tf.float32,
        #         sequence_length=self.length_x,
        #         inputs=out_put_x_1
        #     )
        #     output_fw_x_2, output_bw_x_2 = outputs_x_2
        #     # states_fw_x_2, states_bw_x_2 = states_x_2
        #     out_put_x_2 = (output_fw_x_2 + output_bw_x_2)
        #     outputs_x_2 = tf.multiply(out_put_x_2, self.input_mask_x[:, :, None])
        #     out_put_x = out_put_x_2


# ================================ y, z, p ===========================================================================


        # LSTM layer yz 1 -----------------------------------------------------------------------------------------------
        with tf.variable_scope('lstm_layer_yz_1'):
            cell_f_yz_1 = tf.contrib.rnn.core_rnn_cell.LSTMCell(num_units=hidden_neural_size, state_is_tuple=True)
            cell_b_yz_1 = tf.contrib.rnn.core_rnn_cell.LSTMCell(num_units=hidden_neural_size, state_is_tuple=True)
            outputs_yz_1, states_yz_1 = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_f_yz_1,
                cell_bw=cell_b_yz_1,
                dtype=tf.float32,
                sequence_length=self.length_yz,
                inputs=input_yz
            )
            output_fw_yz_1, output_bw_yz_1 = outputs_yz_1
            # states_fw_yz_1, states_bw_yz_1 = states_yz_1
            out_put_yz_1 = (output_fw_yz_1 + output_bw_yz_1)
            # out_put_yz_1 = output_fw_yz_1
            out_put_yz_1 = tf.multiply(out_put_yz_1, self.input_mask_y[:, :, None])


        # LSTM layer p 1 -----------------------------------------------------------------------------------------------
        with tf.variable_scope('lstm_layer_p_1'):
            cell_f_p_1 = tf.contrib.rnn.core_rnn_cell.LSTMCell(num_units=hidden_neural_size, state_is_tuple=True)
            cell_b_p_1 = tf.contrib.rnn.core_rnn_cell.LSTMCell(num_units=hidden_neural_size, state_is_tuple=True)
            outputs_p_1, states_p_1 = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_f_p_1,
                cell_bw=cell_b_p_1,
                dtype=tf.float32,
                sequence_length=self.length_p,
                inputs=input_p
            )
            output_fw_p_1, output_bw_p_1 = outputs_p_1
            # states_fw_p_1, states_bw_p_1 = states_p_1
            out_put_p_1 = (output_fw_p_1 + output_bw_p_1)
            # out_put_p_1 = output_fw_p_1
            out_put_p_1 = tf.multiply(out_put_p_1, self.input_mask_p[:, :, None])






# ================================= attention ==========================================================================

        with tf.name_scope("attention_layer_1"):
            attention_wx_1 = tf.get_variable("attention_wx_1", initializer=tf.random_normal([hidden_neural_size, attention_size], stddev=0.1))
            attention_wyz_1 = tf.get_variable("attention_wyz_1", initializer=tf.random_normal([hidden_neural_size, attention_size], stddev=0.1))
            attention_wp_1 = tf.get_variable("attention_wp_1", initializer=tf.random_normal([hidden_neural_size, attention_size], stddev=0.1))
            attention_b_1 = tf.get_variable("attention_b_1", initializer=tf.random_normal([attention_size], stddev=0.1))
            attention_v_1 = tf.get_variable("attention_v_1", initializer=tf.random_normal([attention_size], stddev=0.1))

            attention_u_1 = tf.nn.tanh(tf.matmul(tf.reshape(out_put_x, [-1, hidden_neural_size]), attention_wx_1) \
                                  + tf.matmul(tf.reshape(out_put_yz_1, [-1, hidden_neural_size]), attention_wyz_1) \
                                  + tf.matmul(tf.reshape(out_put_p_1, [-1, hidden_neural_size]), attention_wp_1) \
                                  + tf.reshape(attention_b_1, [1, -1]))

            attention_uv_1 = tf.matmul(attention_u_1, tf.reshape(attention_v_1, [-1, 1]))
            self.alphas_1 = tf.nn.softmax(tf.reshape(attention_uv_1, shape=[-1, max_len_x]))
            self.alphas_1 = tf.multiply(tf.exp(self.alphas_1), self.input_mask_x)

            # out_put_att_1 = tf.reduce_sum(out_put_x_1 * tf.reshape(max_alphas_1, [-1, max_len_x, 1]), 1)
            out_put_att_1 = out_put_x * tf.reshape(self.alphas_1, [-1, max_len_x, 1])






# ======================= pooling out_put_x_1 ==========================================================================

        # pooling layer x-----------------------------------------------------------------------------------------------
        with tf.name_scope("max_pooling_layer_x"):
            out_put_x_exp = tf.expand_dims(out_put_x, -1)
            out_put_x_pool = tf.nn.max_pool(
                out_put_x_exp[:, 0:max_len_x, :, :],
                ksize=[1, max_len_x, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="out_put_x_1_pool"
            )
            out_put_x_pool = tf.reshape(out_put_x_pool, [-1, hidden_neural_size])


        # pooling layer att_x_1 ----------------------------------------------------------------------------------------
            out_put_att_x_1_exp = tf.expand_dims(out_put_att_1, -1)
            out_put_att_x_1_pool = tf.nn.max_pool(
                out_put_att_x_1_exp[:, 0:max_len_x, :, :],
                ksize=[1, max_len_x, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="out_put_x_1_pool"
            )
            out_put_att_x_1_pool = tf.reshape(out_put_att_x_1_pool, [-1, hidden_neural_size])



# ==================================s sum ouput ========================================================================


        # sum output layer ---------------------------------------------------------------------------------------------
        with tf.name_scope("sum_output_layer"):
            # out_put = out_put_x_pool


            # out_put_att_x_1_pool = (out_put_att_x_1_pool + out_put_x_pool)
            # out_put = tf.concat([out_put_att_x_1_pool, out_put_x_pool], 1)
            # hidden_neural_size = hidden_neural_size * 2


            out_put = tf.concat([out_put_x_pool, out_put_att_x_1_pool], 1)
            hidden_neural_size = hidden_neural_size * 2


            # out_put = (out_put_att_x_1_pool + out_put_x_pool)




        # # LSTM layer z -------------------------------------------------------------------------------------------------
        # with tf.variable_scope('LSTM_layer_z'):
        #     cell_f_z = tf.nn.rnn_cell.LSTMCell(num_units=hidden_neural_size, state_is_tuple=True)
        #     cell_b_z = tf.nn.rnn_cell.LSTMCell(num_units=hidden_neural_size, state_is_tuple=True)
        #     outputs_z, states_z = tf.nn.bidirectional_dynamic_rnn(
        #         cell_fw=cell_f_z,
        #         cell_bw=cell_b_z,
        #         dtype=tf.float32,
        #         sequence_length=self.length_z,
        #         inputs=input_z
        #     )
        #     output_fw_z, output_bw_z = outputs_z
        #     # states_fw_z, states_bw_z = states_z
        #     out_put_z = (output_fw_z + output_bw_z)



        # # element-wise attention layer ---------------------------------------------------------------------------------
        # with tf.name_scope("element_wise_attention_layer"):
        #     attention_size = hidden_neural_size
        #     out_put_x_tp = tf.transpose(out_put_x, perm=[0, 2, 1])
        #     out_put_y_tp = tf.transpose(out_put_y, perm=[0, 2, 1])
        #     out_put_z_tp = tf.transpose(out_put_z, perm=[0, 2, 1])
        #     attention_wx = tf.get_variable("attention_wx", initializer=tf.random_normal([max_len_x, attention_size], stddev=0.1))
        #     attention_wy = tf.get_variable("attention_wy", initializer=tf.random_normal([max_len_y, attention_size], stddev=0.1))
        #     attention_wz = tf.get_variable("attention_wz", initializer=tf.random_normal([max_len_z, attention_size], stddev=0.1))
        #     attention_b = tf.get_variable("attention_b", initializer=tf.random_normal([attention_size], stddev=0.1))
        #     attention_v = tf.get_variable("attention_v", initializer=tf.random_normal([attention_size], stddev=0.1))
        #
        #     # attention_u = tf.nn.tanh(tf.matmul(tf.reshape(out_put_x_tp, [-1, max_len]), attention_wx) + tf.matmul(tf.reshape(out_put_y, [-1, max_len]), attention_wy) + tf.reshape(attention_b, [1, -1]))
        #     attention_u = tf.nn.tanh(tf.matmul(tf.reshape(out_put_x_tp, [-1, max_len_x]), attention_wx) + tf.matmul(tf.reshape(out_put_y_tp, [-1, max_len_y]), attention_wy) + tf.matmul(tf.reshape(out_put_z_tp, [-1, max_len_z]), attention_wz) + tf.reshape(attention_b, [1, -1]))
        #     # attention_u = tf.nn.tanh(tf.matmul(tf.reshape(out_put_y, [-1, max_len_y]), attention_wy) + tf.reshape(attention_b, [1, -1]))
        #     attention_uv = tf.matmul(attention_u, tf.reshape(attention_v, [-1, 1]))
        #     alphas = tf.nn.softmax(tf.reshape(attention_uv, shape=[-1, hidden_neural_size]))
        #     out_put_y_attention = tf.mul(out_put_y, tf.reshape(alphas, [-1, 1, hidden_neural_size]))
        #     # out_put_y_attention = tf.reduce_sum(out_put_y_attention, 1)
        #     out_put_z_attention = tf.mul(out_put_z, tf.reshape(alphas, [-1, 1, hidden_neural_size]))
        #     # out_put_z_attention = tf.reduce_sum(out_put_z_attention, 1)


        # out_put_x = out_put_x + out_put_y_attention


        # # pooling layer y attention-----------------------------------------------------------------------------------------------
        # with tf.name_scope("max_pooling_layer_y_attention"):
        #     out_put_y_attention = tf.expand_dims(out_put_y_attention, -1)
        #     out_put_y_attention_pool = tf.nn.max_pool(
        #         out_put_y_attention[:, 0:max_len_y, :, :],
        #         ksize=[1, max_len_y, 1, 1],
        #         strides=[1, 1, 1, 1],
        #         padding='VALID',
        #         name="out_put_y_pool"
        #     )
        #     out_put_y_attention = tf.reshape(out_put_y_attention_pool, [-1, hidden_neural_size])
        #
        #


        # # sum output layer ---------------------------------------------------------------------------------------------
        # with tf.name_scope("sum_out_put_layer"):
        #     # out_put = out_put_x + out_put_y_attention + out_put_z_attention
        #     out_put =  tf.concat(1, [out_put_x, out_put_y])
        #     hidden_neural_size = hidden_neural_size * 2



        # dropout layer ------------------------------------------------------------------------------------------------
        with tf.name_scope("dropout_layer"):
            out_put = tf.nn.dropout(out_put, self.keep_prob)
            out_put = tf.nn.tanh(out_put)



        # softmax layer ------------------------------------------------------------------------------------------------
        with tf.name_scope("softmax_layer_bi"):
            self.softmax_w_bi = tf.Variable(tf.truncated_normal([hidden_neural_size, 2], stddev=0.1), name="softmax_w_bi")
            self.softmax_b_bi = tf.Variable(tf.constant(0.1, shape=[2]), name="softmax_b_bi")
            self.logits_bi = tf.nn.xw_plus_b(out_put, self.softmax_w_bi, self.softmax_b_bi, name="logits_bi")


        # softmax layer ------------------------------------------------------------------------------------------------
        with tf.name_scope("softmax_layer"):
            self.softmax_w = tf.Variable(tf.truncated_normal([hidden_neural_size, class_num], stddev=0.1),name="softmax_w")
            self.softmax_b = tf.Variable(tf.constant(0.1, shape=[class_num]), name="softmax_b")
            self.logits = tf.nn.xw_plus_b(out_put, self.softmax_w, self.softmax_b, name="logits")



        # loss layer ---------------------------------------------------------------------------------------------------
        with tf.name_scope("loss_layer"):


            l2_loss = tf.Variable(0.0, name="l2_loss", trainable=True)

            l2_loss += tf.nn.l2_loss(attention_wx_1)
            l2_loss += tf.nn.l2_loss(attention_wyz_1)
            l2_loss += tf.nn.l2_loss(attention_wp_1)
            l2_loss += tf.nn.l2_loss(attention_b_1)
            l2_loss += tf.nn.l2_loss(attention_v_1)


            l2_loss += tf.nn.l2_loss(self.softmax_w_bi)
            l2_loss += tf.nn.l2_loss(self.softmax_b_bi)
            self.cross_entropys_bi = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_label_bi, logits=self.logits_bi + 1e-10)
            self.loss_bi = tf.reduce_mean(self.cross_entropys_bi)


            l2_loss += tf.nn.l2_loss(self.softmax_w)
            l2_loss += tf.nn.l2_loss(self.softmax_b)
            self.cross_entropys = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_label, logits=self.logits + 1e-10)
            self.loss = tf.reduce_mean(self.cross_entropys)

            # self.cross_entropys = fun.softmax_cross_entropy_with_logits(labels=self.input_label, logits=self.logits + 1e-10)

            # self.input_label = math_ops.cast(self.input_label, dtypes.float32)
            # self.loss2 = tf.reduce_sum(tf.square(tf.subtract(self.input_label, self.logits)))
            # self.loss = -tf.reduce_sum(self.input_label * tf.log(self.logits+1e-8) + (1 - self.input_label) * tf.log((1 - self.logits) + 1e-8))
            # self.loss2 = tf.reduce_sum(self.input_label * tf.nn.relu(self.logits), 1)
            # self.loss = -tf.reduce_sum(self.input_label * tf.log(tf.clip_by_value(self.logits, 1e-10, 5.0)), 1)
            # self.loss2 = -tf.arg_max(self.input_label, 1) * tf.log(tf.arg_max(self.logits, 1))


            # self.a = tf.Variable(tf.constant(0.1, shape=[1]), name="a")

            self.loss_plus_l2 = self.loss_bi + self.loss + self.lr * l2_loss
            # self.loss_plus_l2 = self.loss + self.lr * l2_loss

            # if self.global_step <= 125:
            #     self.loss_plus_l2 = self.loss_bi + self.lr * l2_loss
            # else:
            #     self.loss_plus_l2 = self.loss_bi + self.loss + self.lr * l2_loss




        # prediction and accuracy layer --------------------------------------------------------------------------------
        with tf.name_scope("accuracy_layer"):
            self.prediction = tf.argmax(self.logits, 1)
            self.label = tf.arg_max(self.input_label, 1)
            correct_prediction = tf.equal(self.prediction, self.label)
            self.correct_num = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")


        # gradients layer ----------------------------------------------------------------------------------------------
        with tf.name_scope("gradients_layer"):

            # tvars = tf.trainable_variables()
            # grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss_plus_l2, tvars), config.max_grad_norm)
            # # optimizer = tf.train.GradientDescentOptimizer(self.lr)
            # optimizer = tf.train.AdamOptimizer(self.lr)
            # self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

            self.optimizer = tf.train.AdamOptimizer(self.lr)
            self.grads_and_vars = self.optimizer.compute_gradients(self.loss_plus_l2)
            self.train_op = self.optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)




        # summary ------------------------------------------------------------------------------------------------------
        with tf.name_scope("01_train"):
            loss_summary = tf.summary.scalar("01_loss", self.loss)
            loss_plus_l2_summary = tf.summary.scalar("02_loss_plus_l2", self.loss_plus_l2)
            accuracy_summary = tf.summary.scalar("03_accuracy", self.accuracy)
            softmax_w_histogram = tf.summary.histogram("04_softmax_w", self.softmax_w)
            softmax_b_histogram = tf.summary.histogram("05_softmax_b", self.softmax_b)
            alphas_1_histogram = tf.summary.histogram("06_alphas_1", self.alphas_1)


        summary_list = [loss_summary, loss_plus_l2_summary, accuracy_summary, alphas_1_histogram, softmax_w_histogram, softmax_b_histogram]
        self.summary = tf.summary.merge(summary_list)





        # assign new lr
        self.new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self.lr, self.new_lr)

        # assign new cur_num_epoch
        self.new_cur_num_epoch = tf.placeholder(tf.int32, shape=[], name="new_cur_num_epoch")
        self._cur_num_epoch_update = tf.assign(self.cur_num_epoch, self.new_cur_num_epoch)



    def assign_new_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self.new_lr: lr_value})

    def assign_new_cur_num_epoch(self, session, cur_num_epoch):
        session.run(self._cur_num_epoch_update, feed_dict={self.new_cur_num_epoch: cur_num_epoch})
