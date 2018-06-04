import pickle as pkl
import numpy as np
import collections
import random
import os
import shutil
# import gensim
import sys

fix_len = 70

def readData(config, vocab_label, sort_by_len=False):
    print ("\nloading data from ", config.dataset_file, "...")
    with open(config.dataset_file, "rb") as handle:
        train_set = pkl.load(handle)
        test_set = pkl.load(handle)

    # # 1-sent_blind
    # # 2-relation
    # # 3-word_path_string
    # # 4-deprel_path_string
    # # 5-pos_path_string
    # # 6-tree_line
    # # 7-tree_pos_line
    # # 8-sent_id
    # # 9-sent
    # # 10-entities
    # tr_list1, tr_list2, tr_list3, tr_list4, tr_list5, tr_list6, tr_list7, tr_list8, tr_list9, tr_list10, tr_list11, tr_list12 = train_set
    # te_list1, te_list2, te_list3, te_list4, te_list5, te_list6, te_list7, te_list8, te_list9, te_list10, te_list11, te_list12 = test_set
    # handle.close()
    #
    # train_x = tr_list1
    # train_y = tr_list11
    # train_z = tr_list12
    # train_m = tr_list6
    # train_label = tr_list2
    #
    # test_x = te_list1
    # test_y = te_list11
    # test_z = te_list12
    # test_m = te_list6
    # test_label = te_list2



    # 1-sent_blind
    # 2-sent_blind_pos
    # 3-relation
    # 4-dis1
    # 5-dis2
    # 6-sent_id
    # 7-sent
    # 8-entities
    tr_list1, tr_list2, tr_list3, tr_list4, tr_list5, tr_list6, tr_list7, tr_list8 = train_set
    te_list1, te_list2, te_list3, te_list4, te_list5, te_list6, te_list7, te_list8 = test_set
    src_test_set = test_set
    handle.close()

    train_x = tr_list1
    train_y = tr_list4
    train_z = tr_list5
    train_p = tr_list2
    train_label = tr_list3

    test_x = te_list1
    test_y = te_list4
    test_z = te_list5
    test_p = te_list2
    test_label = te_list3


    # split train data into train set and valid set -------------------------------------------------------------------
    # train_set length
    train_num = len(train_x)
    # shuffle and generate train and valid dataset
    sidx = np.random.permutation(train_num)
    n_train = int(np.round(train_num * (1. - config.valid_portion)))
    valid_x = [train_x[s] for s in sidx[n_train:]]
    valid_y = [train_y[s] for s in sidx[n_train:]]
    valid_z = [train_z[s] for s in sidx[n_train:]]
    valid_p = [train_p[s] for s in sidx[n_train:]]
    valid_label = [train_label[s] for s in sidx[n_train:]]
    train_x = [train_x[s] for s in sidx[:n_train]]
    train_y = [train_y[s] for s in sidx[:n_train]]
    train_z = [train_z[s] for s in sidx[:n_train]]
    train_p = [train_p[s] for s in sidx[:n_train]]
    train_label = [train_label[s] for s in sidx[:n_train]]

    print ("data size: all // false // advise // mechanism // effect // int")
    print ("train y: ", len(train_label), " ", train_label.count("false"), " ", train_label.count("advise"), " ", train_label.count("mechanism"), " ", train_label.count("effect"), " ", train_label.count("int"))
    print ("valid y: ", len(valid_label), " ", valid_label.count("false"), " ", valid_label.count("advise"), " ", valid_label.count( "mechanism"), " ", valid_label.count("effect"), " ", valid_label.count("int"))
    print ("test y: ", len(test_label), " ", test_label.count("false"), " ", test_label.count("advise"), " ", test_label.count("mechanism"), " ", test_label.count("effect"), " ", test_label.count("int"))

    # find the length and get max length -------------------------------------------------------------------------------
    train_x_len, valid_x_len, test_x_len = findLength([train_x, valid_x, test_x])
    train_y_len, valid_y_len, test_y_len = findLength([train_y, valid_y, test_y])
    train_z_len, valid_z_len, test_z_len = findLength([train_z, valid_z, test_z])
    train_p_len, valid_p_len, test_p_len = findLength([train_p, valid_p, test_p])
    # if the value of "valid_portion" is 0
    if len(valid_x_len) ==0 or len(valid_y_len) ==0 or len(valid_z_len) ==0 or len(valid_p_len) ==0:
        max_len_x = max([max(train_x_len), max(test_x_len)])
        max_len_y = max([max(train_y_len), max(test_y_len)])
        max_len_z = max([max(train_z_len), max(test_z_len)])
        max_len_p = max([max(train_p_len), max(test_p_len)])
    else:
        max_len_x = max([max(train_x_len), max(valid_x_len), max(test_x_len)])
        max_len_y = max([max(train_y_len), max(valid_y_len), max(test_y_len)])
        max_len_z = max([max(train_z_len), max(valid_z_len), max(test_z_len)])
        max_len_p = max([max(train_p_len), max(valid_p_len), max(test_p_len)])
    max_len_list = [max_len_x, max_len_y, max_len_z, max_len_p]
    print ("max len x, y, z, p: ", max_len_x, " ", max_len_y, " ", max_len_z, " ", max_len_p)

    # build vocabulary -------------------------------------------------------------------------------------------------
    vocab_x = buildVocabulary([train_x, valid_x, test_x], config.vocab_limit)
    vocab_y = buildVocabulary([train_y, valid_y, test_y], config.vocab_limit)
    vocab_z = buildVocabulary([train_z, valid_z, test_z], config.vocab_limit)
    vocab_p = buildVocabulary([train_p, valid_p, test_p], config.vocab_limit)
    # vocab_yz = dict(vocab_y.items() + vocab_z.items())
    vocab_yz = vocab_y.copy()
    vocab_yz.update(vocab_z)
    vocab_size_x = len(vocab_x)
    vocab_size_y = len(vocab_y)
    vocab_size_z = len(vocab_z)
    vocab_size_yz = len(vocab_yz)
    vocab_size_p = len(vocab_p)
    print ("vocab_size_x: ", vocab_size_x)
    print ("vocab_size_y: ", vocab_size_y)
    print ("vocab_size_z: ", vocab_size_z)
    print ("vocab_size_yz: ", vocab_size_yz)
    print ("vocab_size_p: ", vocab_size_p)

    with open("./vocabulary/vocab_x_yz_p.pickle", "wb") as handle:
        pkl.dump(vocab_x, handle)
        pkl.dump(vocab_yz, handle)
        pkl.dump(vocab_p, handle)


    # # making glove embedding for x vocab -------------------------------------------------------------------------------
    # # glove_model = gensim.models.KeyedVectors.load_word2vec_format("./glove/glove_model.txt")
    # glove_embed_vocab_x = []
    # for key, value in vocab_x.iteritems():
    #     # if key in glove_model:
    #     #     glove_embed_vocab_x.append(glove_model["key"])
    #     # else:
    #     #     glove_embed_vocab_x.append(-1+2*np.random.random(size=embedding_dim))
    #     glove_embed_vocab_x.append(-1 + 2 * np.random.random(size=embedding_dim))
    # glove_embed_vocab_y = []
    # for key, value in vocab_y.iteritems():
    #     glove_embed_vocab_y.append(-1 + 2 * np.random.random(size=distance_embedding_dim))
    # glove_embed_vocab_z = []
    # for key, value in vocab_z.iteritems():
    #     glove_embed_vocab_z.append(-1 + 2 * np.random.random(size=distance_embedding_dim))
    # glove_embed_vocab_yz = []
    # for key, value in vocab_yz.iteritems():
    #     glove_embed_vocab_yz.append(-1 + 2 * np.random.random(size=distance_embedding_dim))
    #
    # with open("./vocab_glove_embed/vocab_xyz_bemdding.txt", "wb") as handle:
    #     pkl.dump(glove_embed_vocab_x, handle)
    #     pkl.dump(glove_embed_vocab_y, handle)
    #     pkl.dump(glove_embed_vocab_z, handle)
    #     pkl.dump(glove_embed_vocab_yz, handle)




    # mapping data into index ------------------------------------------------------------------------------------------
    train_x_index, valid_x_index, test_x_index = mapTermToIndex([train_x, valid_x, test_x], vocab_x)
    train_y_index, valid_y_index, test_y_index = mapTermToIndex([train_y, valid_y, test_y], vocab_yz)
    train_z_index, valid_z_index, test_z_index = mapTermToIndex([train_z, valid_z, test_z], vocab_yz)
    train_p_index, valid_p_index, test_p_index = mapTermToIndex([train_p, valid_p, test_p], vocab_p)
    train_label_index, valid_label_index, test_label_index = mapLabelToIndex([train_label, valid_label, test_label], vocab_label)
    # print (train_x_index)

    # padding data to max len and turning label into vector ------------------------------------------------------------
    train_x, valid_x, test_x, mask_train_x, mask_valid_x, mask_test_x = padding([train_x_index, valid_x_index, test_x_index], max_len_x)
    train_y, valid_y, test_y, mask_train_y, mask_valid_y, mask_test_y  = padding([train_y_index, valid_y_index, test_y_index], max_len_y)
    train_z, valid_z, test_z, mask_train_z, mask_valid_z, mask_test_z  = padding([train_z_index, valid_z_index, test_z_index], max_len_z)
    train_p, valid_p, test_p, mask_train_p, mask_valid_p, mask_test_p  = padding([train_p_index, valid_p_index, test_p_index], max_len_p)
    train_label, valid_label, test_label = paddingLabelToVector([train_label_index, valid_label_index, test_label_index], len(vocab_label))
    train_label_bi, valid_label_bi, test_label_bi = getBinaryLabelVector([train_label, valid_label, test_label])

    # return set -------------------------------------------------------------------------------------------------------
    train_set = (train_x, train_y, train_z, train_p, train_label, train_label_bi, train_x_len, train_y_len, train_z_len, train_p_len)
    valid_set = (valid_x, valid_y, valid_z, valid_p, valid_label, valid_label_bi, valid_x_len, valid_y_len, valid_z_len, valid_p_len)
    test_set = (test_x, test_y, test_z, test_p, test_label, test_label_bi, test_x_len, test_y_len, test_z_len, test_p_len)

    mask_train_set = (mask_train_x, mask_train_y, mask_train_z, mask_train_p)
    mask_valid_set = (mask_valid_x, mask_valid_y, mask_valid_z, mask_valid_p)
    mask_test_set = (mask_test_x, mask_test_y, mask_test_z, mask_test_p)


    data_set = (train_set, valid_set, test_set, mask_train_set, mask_valid_set, mask_test_set)
    paras = (max_len_list, vocab_size_x, vocab_size_y, vocab_size_z, vocab_size_yz, vocab_size_p)

    return data_set, paras, src_test_set


def findLength(lists):
    lists_len = []
    for list in lists:
        lists_len.append([len(l) for l in list])
    return lists_len


# def findLength(lists):
#     lists_len = []
#     for list in lists:
#         list_len = []
#         for l in list:
#             if len(l)>fix_len:
#                 list_len.append(fix_len)
#             else:
#                 list_len.append(len(l))
#         lists_len.append(list_len)
#     return lists_len


def buildVocabulary(lists, vocab_limit):
    vocabulary = {}
    for list in lists:
        for l in list:
            for term in l:
                if term in vocabulary:
                    vocabulary[term] += 1
                else:
                    vocabulary[term] = 0

    vocab = collections.OrderedDict()
    i = 1
    vocab['UNK'] = 0
    for term, count in vocabulary.items():
        if count >= vocab_limit:
            vocab[term] = i
            i += 1
    return vocab


def mapTermToIndex(lists, vocabulary):
    lists_index = []
    for list in lists:
        # lists_index.append([[vocabulary[term] for term in l] for l in list])

        list_index = []
        for l in list:
            temp = []
            for term in l:
                if term in vocabulary:
                    temp.append(vocabulary[term])
                else:
                    temp.append(0)
            list_index.append(temp)
        lists_index.append(list_index)

    return lists_index


def mapLabelToIndex(lists, vocabulary):
    lists_index = []
    for list in lists:
        lists_index.append([vocabulary[term] for term in list])
    return lists_index


def padding(lists, max_len):

    lists_padding = []
    mask_lists = []
    for list in lists:
        list_padding = []
        mask_list = np.zeros([len(list), max_len])
        for l, j in zip(list, range(len(list))):
            length = len(l)
            l_padding = []
            for i in range(length):
                l_padding.append(l[i])
                mask_list[j][i] = 1
            for i in range(length, max_len):
                l_padding.append(0)
            list_padding.append(l_padding)

        lists_padding.append(list_padding)
        mask_lists.append(mask_list)
    return_lists = lists_padding + mask_lists
    return  return_lists


# def padding(lists, max_len):
#     lists_padding = []
#     for list in lists:
#         list_padding = []
#         for l in list:
#             length = len(l)
#             l_padding = []
#             if length > max_len:
#                 for i in range(max_len):
#                     l_padding.append(l[i])
#             else:
#                 for i in range(length):
#                     l_padding.append(l[i])
#                 for i in range(length, max_len):
#                     l_padding.append(0)
#             list_padding.append(l_padding)
#         lists_padding.append(list_padding)
#     return  lists_padding



def paddingLabelToVector(lists, vector_size):
    lists_vector = []
    for list in lists:
        list_vector = np.zeros((len(list), vector_size))
        for i in range(len(list)):
            list_vector[i][list[i]] = 1.0
        lists_vector.append(list_vector)
    return lists_vector

def getBinaryLabelVector(lists):
    lists_vector = []
    for list in lists:
        list_vector = []
        for l in list:
            max_index = np.argmax(l)
            # print (max_index)
            if max_index == 0:
                list_vector.append([0.0, 1.0])
            else:
                list_vector.append([1.0, 0.0])
        # print (list_vector)
        lists_vector.append(list_vector)
    return lists_vector


# def mapIndexToTerm(list, vocabulary):
#     return [[vocabulary[term] for term in l] for l in list]


# return batch dataset
def batchIter(data,  batch_size):
    x, y, z, p, label, label_bi, length_x, length_y, length_z, length_p, mask_x, mask_y, mask_z, mask_p = data
    x, y, z, p, label, label_bi, length_x, length_y, length_z, length_p, mask_x, mask_y, mask_z, mask_p = \
        disturbOrder([x, y, z, p, label, label_bi, length_x, length_y, length_z, length_p, mask_x, mask_y, mask_z, mask_p])
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    p = np.array(p)
    mask_x = np.array(mask_x)
    mask_y = np.array(mask_y)
    mask_z = np.array(mask_z)
    mask_p = np.array(mask_p)
    label = np.array(label)
    label_bi = np.array(label_bi)
    length_x = np.array(length_x)
    length_y = np.array(length_y)
    length_z = np.array(length_z)
    length_p = np.array(length_p)
    data_size = len(x)
    num_batches_per_epoch = int((data_size + batch_size - 1) / batch_size)
    for batch_index in range(num_batches_per_epoch):
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, data_size)
        return_x = x[start_index:end_index]
        return_y = y[start_index:end_index]
        return_z = z[start_index:end_index]
        return_p = p[start_index:end_index]
        return_mask_x = mask_x[start_index:end_index]
        return_mask_y = mask_y[start_index:end_index]
        return_mask_z = mask_z[start_index:end_index]
        return_mask_p = mask_p[start_index:end_index]
        return_label = label[start_index:end_index]
        return_label_bi = label_bi[start_index:end_index]
        return_length_x = length_x[start_index:end_index]
        return_length_y = length_y[start_index:end_index]
        return_length_z = length_z[start_index:end_index]
        return_length_p = length_p[start_index:end_index]
        yield (return_x, return_y, return_z, return_p, return_label, return_label_bi, return_length_x, return_length_y, return_length_z, return_length_p, return_mask_x, return_mask_y, return_mask_z, return_mask_p)


def disturbOrder(lists):
    lists_new = []
    length = len(lists[0])
    shuffle_indices = np.random.permutation(np.arange(length))
    for list in lists:
        list = np.array(list)
        lists_new.append(list[shuffle_indices])
    return lists_new


def sampleData(config, data, vocab_label):
    x, y, z, p, label, label_bi, length_x, length_y, length_z, length_m, m_x, m_y, m_z, m_p = data

    index_lists = []
    for key, value in vocab_label.items():
        index_lists.append([i for i in range(len(label)) if list(label[i]).index(max(label[i])) == value])

    # max instances undersampling
    list_len = [len(l) for l in index_lists]
    max_set_index = list_len.index(max(list_len))
    index_lists[max_set_index] = random.sample(index_lists[max_set_index], int(config.sample_ratio * len(index_lists[max_set_index])))
    # min instances oversampling
    # min_set_index = list_len.index(min(list_len))
    # part_one = random.sample(index_lists[min_set_index], int(0.8 * len(index_lists[min_set_index])))
    # part_two = random.sample(index_lists[min_set_index], int(0.8 * len(index_lists[min_set_index])))
    # part_three = random.sample(index_lists[min_set_index], int(0.8 * len(index_lists[min_set_index])))
    # part_four = random.sample(index_lists[min_set_index], int(0.8 * len(index_lists[min_set_index])))
    # index_lists[min_set_index] = index_lists[min_set_index] + part_one + part_two + part_three + part_four

    # list_len[max_set_index] = 0
    # second_max_set_index = list_len.index(max(list_len))
    # index_lists[second_max_set_index] = random.sample(index_lists[second_max_set_index], int(0.6 * len(index_lists[second_max_set_index])))

    new_x = []
    new_y = []
    new_z = []
    new_p = []
    new_mask_x = []
    new_mask_y = []
    new_mask_z = []
    new_mask_p = []
    new_label = []
    new_label_bi = []
    new_length_x = []
    new_length_y = []
    new_length_z = []
    new_length_p = []
    temp = []
    for index_l in index_lists:
        for index in index_l:
            new_x.append(x[index])
            new_y.append(y[index])
            new_z.append(z[index])
            new_p.append(p[index])
            new_mask_x.append(m_x[index])
            new_mask_y.append(m_y[index])
            new_mask_z.append(m_z[index])
            new_mask_p.append(m_p[index])

            new_label.append(label[index])
            new_label_bi.append(label_bi[index])
            new_length_x.append(length_x[index])
            new_length_y.append(length_y[index])
            new_length_z.append(length_z[index])
            new_length_p.append(length_m[index])
            temp.append(list(label[index]).index(max(label[index])))
    print ("sample train y: ", len(temp), " ", temp.count(0), " ", temp.count(1), " ", temp.count(2), " ", temp.count(3), " ", temp.count(4))

    new_data = (new_x, new_y, new_z, new_p, new_label, new_label_bi, new_length_x, new_length_y, new_length_z, new_length_p, new_mask_x,  new_mask_y, new_mask_z, new_mask_p)
    return new_data



def resetDir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
        os.makedirs(dir)
    else:
        os.makedirs(dir)