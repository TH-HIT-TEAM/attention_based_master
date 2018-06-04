"""
by Weby
g.webywang@gmail.com
2017/11/29

python 3.6
tensorflow 1.0.1
"""

import tensorflow as tf
import os
import data_helper
import time
from model import Model
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import numpy as np
import pickle as pkl
import sys
import shutil

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('num_epoch', 5, 'num epoch')
flags.DEFINE_float('valid_portion', 0, 'valid portion')  # 0.1
flags.DEFINE_integer('batch_size', 100, 'the batch_size of the training procedure')
flags.DEFINE_integer('embedding_dim', 10, 'embedding dim')
flags.DEFINE_integer('pos_embedding_dim', 10, 'pos embedding dim')
flags.DEFINE_integer('distance_embedding_dim', 10, 'distance embedding dim')
flags.DEFINE_integer('hidden_neural_size', 10, 'hidden neural size')
flags.DEFINE_integer('attention_size', 10, 'attention size')
flags.DEFINE_float('lr', 0.01, 'the lsearning rate')
flags.DEFINE_float('sample_ratio', 0.01, 'the sample ratio')
flags.DEFINE_integer('max_decay_epoch', 100, 'num epoch')
flags.DEFINE_float('lr_decay', 0.8, 'the learning rate decay')
flags.DEFINE_float('keep_prob', 0.7, 'dropout rate')
flags.DEFINE_integer('max_grad_norm', 5, 'max grad norm')
flags.DEFINE_boolean('is_summary', False, 'summary or not')
flags.DEFINE_integer('vocab_limit', 0, 'the limiting count to be added in the vocab')
flags.DEFINE_boolean('load_existed_model', False, 'load existed model or not')

flags.DEFINE_string('dataset_file', "./preprocess/ddi_corpus/ddi_corpus_wpd.pickle", 'dataset file')
# flags.DEFINE_string('out_dir', os.path.abspath(os.path.join(os.path.curdir, "runs")), 'output directory')
flags.DEFINE_string('out_dir', "./output", 'output directory')
flags.DEFINE_integer('check_point_every', 10, 'checkpoint every num epoch ')
vocab_label = {'false': 0, 'advise': 1, 'mechanism': 2, 'effect': 3, 'int': 4}
# vocab_label_reverse = {0: 'false', 1: 'advise', 2: 'mechanism', 3: 'effect', 4: 'int'}
vocab_label_reverse = {value: key for key, value in vocab_label.items()}


class Config(object):
    num_epoch = FLAGS.num_epoch
    cur_num_epoch = 0
    class_num = len(vocab_label)
    valid_portion = FLAGS.valid_portion
    batch_size = FLAGS.batch_size
    embedding_dim = FLAGS.embedding_dim
    pos_embedding_dim = FLAGS.pos_embedding_dim
    distance_embedding_dim = FLAGS.distance_embedding_dim
    hidden_neural_size = FLAGS.hidden_neural_size
    attention_size = FLAGS.attention_size
    lr = FLAGS.lr
    sample_ratio = FLAGS.sample_ratio
    max_decay_epoch = FLAGS.max_decay_epoch
    is_summary = FLAGS.is_summary
    vocab_limit = FLAGS.vocab_limit
    lr_decay = FLAGS.lr_decay
    keep_prob = FLAGS.keep_prob
    max_grad_norm = FLAGS.max_grad_norm
    load_existed_model = FLAGS.load_existed_model

    max_len_list = []  # init value
    vocab_size_x = 0  # init value
    vocab_size_y = 0  # init value
    vocab_size_z = 0  # init value
    vocab_size_yz = 0  # init value
    vocab_size_p = 0  # init value
    dataset_file = FLAGS.dataset_file
    out_dir = FLAGS.out_dir
    checkpoint_every = FLAGS.check_point_every


def train_step():
    config = Config()

    # load data --------------------------------------------------------------------------------------------------------
    data_set, paras, src_test_set = data_helper.readData(config, vocab_label)
    train_data, valid_data, test_data, mask_train_data, mask_valid_data, mask_test_data = data_set
    train_data = train_data + mask_train_data
    valid_data = valid_data + mask_valid_data
    test_data = test_data + mask_test_data
    max_len_list, vocab_size_x, vocab_size_y, vocab_size_z, vocab_size_yz, vocab_size_p = paras
    config.max_len_list = max_len_list
    config.vocab_size_x = vocab_size_x
    config.vocab_size_y = vocab_size_y
    config.vocab_size_z = vocab_size_z
    config.vocab_size_yz = vocab_size_yz
    config.vocab_size_p = vocab_size_p



    with tf.Graph().as_default(), tf.Session() as session:
        model = Model(config=config)
        tf.initialize_all_variables().run()

        # # load glove_embed_vocab_x file
        # with open("./vocab_glove_embed/vocab_xyz_bemdding.txt", "rb") as handle:
        #     embed_vocab_x = pkl.load(handle)
        #     embed_vocab_y = pkl.load(handle)
        #     embed_vocab_z = pkl.load(handle)
        #     embed_vocab_yz = pkl.load(handle)
        #     print ("load vocab embddding file ..."

        # begin training
        begin_time = int(time.time())
        print ("begin training ...")

        # save embedding
        embed_dir = "./embedding"
        data_helper.resetDir(embed_dir)
        embedding = session.run(model.embedding)
        with open(embed_dir + "/" + "embed_x_yz_p.pickle", "wb") as handle:
            pkl.dump(embedding, handle)


        # save test data
        test_label_list = []
        test_prediction_list = []
        att_weight_1_list = []

        # save model
        checkpoint_dir = os.path.join(config.out_dir, "checkpoints")
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(max_to_keep=2)
        # saver = tf.train.Saver()


        # load existed model
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if config.load_existed_model and latest_checkpoint:
            print ("load existed model: ", latest_checkpoint, " ...")
            graph = tf.get_default_graph()
            saver.restore(session, latest_checkpoint)
            config.cur_num_epoch = session.run(graph.get_tensor_by_name('cur_num_epoch:0'))



        # summary
        if config.is_summary:
            from summary import Summary
            # train summary
            train_summary_dir = os.path.join(config.out_dir, "log", "train")
            resetDir(train_summary_dir)
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, session.graph)
            # valid summary
            valid_summary_dir = os.path.join(config.out_dir, "log", "valid")
            resetDir(valid_summary_dir)
            valid_summary_writer = tf.summary.FileWriter(valid_summary_dir, session.graph)
            summary_valid = Summary("02_valid")
            #test summary
            test_summary_dir = os.path.join(config.out_dir, "log", "test")
            resetDir(test_summary_dir)
            test_summary_writer = tf.summary.FileWriter(test_summary_dir, session.graph)
            summary_test = Summary("03_test")
        else:
            train_summary_writer = ""
            valid_summary_writer = ""
            summary_valid = ""
            test_summary_writer = ""
            summary_test = ""



        # training
        while config.cur_num_epoch < config.num_epoch:
        # for i in range(config.num_epoch):
        #     config.cur_num_epoch += 1

            lr_decay = config.lr_decay ** max(config.cur_num_epoch - config.max_decay_epoch, 0.0) # ** is exponentiation
            model.assign_new_lr(session, config.lr * lr_decay)

            model.assign_new_cur_num_epoch(session, config.cur_num_epoch)

            print("\nthe %d epoch training, " % (config.cur_num_epoch + 1), ", lr is: ", config.lr * lr_decay)


            # sampling data
            train_data_samples = data_helper.sampleData(config, train_data, vocab_label)
            # train_data_samples = train_data



            # =================================== train ================================================================
            global_step = run_epoch(model, session, config , train_data_samples, valid_data,
                                        train_summary_writer, summary_valid, valid_summary_writer)


            # save model
            # if i % config.checkpoint_every == 0:
            #     path = saver.save(session, checkpoint_prefix, global_step)
            saver.save(session, checkpoint_prefix, global_step=config.cur_num_epoch+1)
            # saver.save(session, checkpoint_prefix, global_step=global_step)



            # ==================================== test ================================================================

            test_accuracy, test_label, test_prediction, att_weight_1 = evaluate(model, session, test_data)
            test_label_list.append(test_label)
            test_prediction_list.append(test_prediction)
            att_weight_1_list.append(att_weight_1)

            presicion = precision_score(test_label, test_prediction, [1, 2, 3, 4], average="micro")
            recall = recall_score(test_label, test_prediction, [1, 2, 3, 4], average="micro")
            f1 = f1_score(test_label, test_prediction, [1, 2, 3, 4], average="micro")

            if config.is_summary:
                # test summary
                feed_dict = {
                    summary_test.precision: presicion,
                    summary_test.recall: recall,
                    summary_test.f1: f1
                }
                test_summary = session.run(summary_test.summary, feed_dict)
                test_summary_writer.add_summary(test_summary, config.cur_num_epoch)
                test_summary_writer.flush()

            print ("==== the test precision is ", presicion, ", the recall is ", recall, ", the f1 is ", f1, "====")

            config.cur_num_epoch += 1


        # ======================================  finish train ==================================

        print ("the train is finished")
        end_time = int(time.time())
        print ("training takes %d seconds already\n" % (end_time - begin_time))

        # get best score
        fscore_list = []
        for test_label, test_prediction in zip(test_label_list, test_prediction_list):
            fscore_list.append(f1_score(test_label, test_prediction, [1, 2, 3, 4], average="micro"))
        best_score_index = np.argmax(fscore_list)
        test_label_best = test_label_list[best_score_index]
        test_prediction_best = test_prediction_list[best_score_index]
        att_weight_1_best = att_weight_1_list[best_score_index]
        print ("\n\nbest fscore epoch: ", best_score_index + 1)
        cnf_matrix = confusion_matrix(test_label_best, test_prediction_best)


        presicion = "P: " + str(precision_score(test_label_best, test_prediction_best, [1, 2, 3, 4], average="micro"))
        recall = "R: " + str(recall_score(test_label_best, test_prediction_best, [1, 2, 3, 4], average="micro"))
        f1 = "F: " + str(f1_score(test_label_best, test_prediction_best, [1, 2, 3, 4], average="micro"))
        print ("======= best test result: ", presicion + "\t" + recall + "\t" + f1, "==========")
        print (cnf_matrix)
        # print("======= best attention weight: ", att_weight_1_best, "=========")
        # print("======= best attention count ", len(att_weight_1_best), "  " , len(att_weight_1_best[0]), "=========")




        # save best test result
        best_result_dir = os.path.join(config.out_dir, "best_result")
        if not os.path.exists(best_result_dir):
            os.makedirs(best_result_dir)

        w_test_result = open(best_result_dir + "/test_result.txt", "w")
        te_list1, te_list2, te_list3, te_list4, te_list5, te_list6, te_list7, te_list8 = src_test_set

        for term1, term2, term3, term4, term5, term6, term7, term8, pred, att_weight_1 in zip(te_list1, te_list2, te_list3, te_list4, te_list5, te_list6, te_list7, te_list8, test_prediction_best, att_weight_1_best):
            w_test_result.write(term6 + "\n")
            w_test_result.write(" ".join(term7) + "\n")
            w_test_result.write(term8 + "\n")
            w_test_result.write(" ".join(term1) + "\n")
            w_test_result.write(" ".join(term2) + "\n")
            w_test_result.write(" ".join(term4) + "\n")
            w_test_result.write(" ".join(term5) + "\n")
            w_test_result.write(term3 + "\n")
            w_test_result.write(vocab_label_reverse[pred] + "\n")
            w_test_result.write(" ".join([str(i) for i in att_weight_1]) + "\n")
            w_test_result.write(" ".join([str(i) for i in softmax(att_weight_1[:len(term1)])]) + "\n")
            w_test_result.write("\n")

        rw = open(best_result_dir + "/class_prf.txt", "w")
        rw.write("the training takes " + str(end_time - begin_time) + " s")
        rw.write("weighted: ")
        rw.write(str(precision_score(test_label_best, test_prediction_best, [1, 2, 3, 4], average="weighted")) + "\t")
        rw.write(str(recall_score(test_label_best, test_prediction_best, [1, 2, 3, 4], average="weighted")) + "\t")
        rw.write(str(f1_score(test_label_best, test_prediction_best, [1, 2, 3, 4], average="weighted")) + "\t")
        rw.write("\n")

        rw.write("micro: ")
        rw.write(str(precision_score(test_label_best, test_prediction_best, [1, 2, 3, 4], average="micro")) + "\t")
        rw.write(str(recall_score(test_label_best, test_prediction_best, [1, 2, 3, 4], average="micro")) + "\t")
        rw.write(str(f1_score(test_label_best, test_prediction_best, [1, 2, 3, 4], average="micro")) + "\t")
        rw.write("\n")

        rw.write("macro: ")
        rw.write(str(precision_score(test_label_best, test_prediction_best, [1, 2, 3, 4], average="macro")) + "\t")
        rw.write(str(recall_score(test_label_best, test_prediction_best, [1, 2, 3, 4], average="macro")) + "\t")
        rw.write(str(f1_score(test_label_best, test_prediction_best, [1, 2, 3, 4], average="macro")) + "\t")
        rw.write("\n")

        rw.write("1: ")
        rw.write(str(precision_score(test_label_best, test_prediction_best, [1], average="micro")) + "\t")
        rw.write(str(recall_score(test_label_best, test_prediction_best, [1], average="micro")) + "\t")
        rw.write(str(f1_score(test_label_best, test_prediction_best, [1], average="micro")) + "\t")
        rw.write("\n")

        rw.write("2: ")
        rw.write(str(precision_score(test_label_best, test_prediction_best, [2], average="micro")) + "\t")
        rw.write(str(recall_score(test_label_best, test_prediction_best, [2], average="micro")) + "\t")
        rw.write(str(f1_score(test_label_best, test_prediction_best, [2], average="micro")) + "\t")
        rw.write("\n")

        rw.write("3: ")
        rw.write(str(precision_score(test_label_best, test_prediction_best, [3], average="micro")) + "\t")
        rw.write(str(recall_score(test_label_best, test_prediction_best, [3], average="micro")) + "\t")
        rw.write(str(f1_score(test_label_best, test_prediction_best, [3], average="micro")) + "\t")
        rw.write("\n")

        rw.write("4: ")
        rw.write(str(precision_score(test_label_best, test_prediction_best, [4], average="micro")) + "\t")
        rw.write(str(recall_score(test_label_best, test_prediction_best, [4], average="micro")) + "\t")
        rw.write(str(f1_score(test_label_best, test_prediction_best, [4], average="micro")) + "\t")
        rw.write("\n")

        rw.write(str(cnf_matrix) + "\n")


        if config.is_summary:
            # close summary writer
            train_summary_writer.close()
            valid_summary_writer.close()
            test_summary_writer.close()




def run_epoch(model, session, config, train_data, valid_data, train_summary_writer, summary_valid, vaild_summary_writer):
    for step, (x, y, z, p, label, label_bi, length_x, length_y, length_z, length_p, m_x, m_y, m_z, m_p) in enumerate(data_helper.batchIter(train_data, batch_size=FLAGS.batch_size)):
        feed_dict = {
            model.input_x: x,
            model.input_y: y,
            model.input_z: z,
            model.input_p: p,
            model.input_mask_x: m_x,
            model.input_mask_y: m_y,
            model.input_mask_z: m_z,
            model.input_mask_p: m_p,
            model.input_label: label,
            model.input_label_bi: label_bi,
            model.length_x: length_x,
            model.length_y: length_y,
            model.length_z: length_z,
            model.length_p: length_p,
            model.keep_prob: config.keep_prob,
        }

        fetches = [model.train_op, model.global_step, model.loss, model.loss_plus_l2, model.accuracy, model.summary]
        _, global_step, loss, loss_plus_l2, accuracy, summary = session.run(fetches, feed_dict)

        if config.is_summary:
            # add train summary
            train_summary_writer.add_summary(summary, global_step)
            train_summary_writer.flush()

        print ("the %i step, train loss_entropy is %f, loss_plus_l2 is: %f, accuracy is %f" % (global_step, loss, loss_plus_l2, accuracy))


        # if (global_step % 15 == 0):
        #     valid_accuracy, valid_label, valid_prediction, _ = evaluate(model, session, valid_data)
        #     presicion = precision_score(valid_label, valid_prediction, [1, 2, 3, 4], average="micro")
        #     recall = recall_score(valid_label, valid_prediction, [1, 2, 3, 4], average="micro")
        #     f1 = f1_score(valid_label, valid_prediction, [1, 2, 3, 4], average="micro")
        #
        #     # valid summary
        #     feed_dict = {
        #         summary_valid.precision: presicion,
        #         summary_valid.recall: recall,
        #         summary_valid.f1: f1
        #     }
        #     test_summary = session.run(summary_valid.summary, feed_dict)
        #     vaild_summary_writer.add_summary(test_summary, global_step)
        #     vaild_summary_writer.flush()

    return global_step





def evaluate(model, session, data):
    correct_num = 0
    predictions = []
    labels = []
    att_weight_1s = []
    total_num = len(data[0])
    for step, (x, y, z, p, label, label_bi, length_x, length_y, length_z, length_p, m_x, m_y, m_z, m_p) in enumerate(data_helper.batchIter(data, batch_size=FLAGS.batch_size)):
        feed_dict = {
            model.input_x: x,
            model.input_y: y,
            model.input_z: z,
            model.input_p: p,
            model.input_mask_x: m_x,
            model.input_mask_y: m_y,
            model.input_mask_z: m_z,
            model.input_mask_p: m_p,
            model.input_label: label,
            model.input_label_bi: label_bi,
            model.length_x: length_x,
            model.length_y: length_y,
            model.length_z: length_z,
            model.length_p: length_p,
            model.keep_prob: 1.0,
        }
        fetches = [model.correct_num, model.label, model.prediction, model.alphas_1]
        count, label, pred, att_weight_1 = session.run(fetches, feed_dict)
        correct_num += count
        predictions += list(pred)
        labels += list(label)
        att_weight_1s += list(att_weight_1)
    accuracy = float(correct_num) / total_num

    return accuracy, labels, predictions, att_weight_1s



def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x



def resetDir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
        os.makedirs(dir)
    else:
        os.makedirs(dir)


def main(_):
    train_step()

if __name__ == "__main__":
    tf.app.run()









