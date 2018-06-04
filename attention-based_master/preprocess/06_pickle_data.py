import pickle as pkl
import re

def dataRead(path):
    print ("processing: ", path)
    file = open(path, "r")
    instances = file.read().strip().split('\n\n')
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    list5 = []
    list6 = []
    list7 = []
    list8 = []
    list9 = []
    list10 = []
    list11 = []
    list12 = []
    for instance in instances:
        lines = instance.strip().split("\n")

        # ------------------------------------------------

        # value1 = lines[5].lower()
        # value2 = lines[1].lower()
        # value3 = lines[6].lower()
        #
        # # value1 = re.sub("e.g\s\.", "e.g.", value1)  # "e.g ." to "e.g."
        # # value1 = re.sub("i.e\s\.", "e.g.", value1)  # "i.e ." to "i.e."
        # value1 = re.sub("\s\.", "", value1)  # delete " ."
        # value1 = re.sub("\(", " ( ", value1)  # delete " ."
        # value1 = re.sub("\)", " ) ", value1)  # delete " ."
        # value1 = re.sub("drug_n", " drug_n ", value1)  # "drug_n14" to "drug_n 14"
        # value1 = re.sub("drug_1", " drug_1 ", value1)  # "drug_114" to "drug_1 14"
        # value1 = re.sub("drug_2", " drug_2 ", value1)  # "drug_214" to "drug_2 14"
        # value1 = re.sub("\d+\,\d+\s", " TAG_OF_DIGIT ", value1)  # "drugs1,3" to "drugs 1,3"
        #
        # value3 = re.sub("\prp\s\$", "prp$", value3)  # "prp $" to "prp$", "wp $" to "wp$"
        # value3 = re.sub("\wp\s\$", "wp$", value3)  # "prp $" to "prp$", "wp $" to "wp$"


    # ----------------------------------------------------

    #     value1 = lines[0].lower()
    #     value2 = lines[1].lower()
    #     value3 = lines[2].lower()
    #     value4 = lines[3].lower()
    #     value5 = lines[4].lower()
    #     value6 = lines[5].lower()
    #     value7 = lines[6].lower()
    #     value8 = lines[7].lower()
    #     value9 = lines[8].lower()
    #     value10 = lines[9].lower()
    #
    #
    #     value6 = re.sub("\s\.", "", value6)  # delete " ."
    #     value6 = re.sub("\(", " ( ", value6)  # delete " ("
    #     value6 = re.sub("\)", " ) ", value6)  # delete " )"
    #     value6 = re.sub("drug_n", " drug_n ", value6)  # "drug_n14" to "drug_n 14"
    #     value6 = re.sub("drug_1", " drug_1 ", value6)  # "drug_114" to "drug_1 14"
    #     value6 = re.sub("drug_2", " drug_2 ", value6)  # "drug_214" to "drug_2 14"
    #     value6 = re.sub("\d+\,\d+\s", " TAG_OF_DIGIT ", value6)  # "drugs1,3" to "drugs 1,3"
    #
    #     value7 = re.sub("\prp\s\$", "prp$", value7)  # "prp $" to "prp$", "wp $" to "wp$"
    #     value7 = re.sub("\wp\s\$", "wp$", value7)  # "prp $" to "prp$", "wp $" to "wp$"
    #
    #
    #     list1.append(value1.split())
    #     list2.append(value2)
    #     list3.append(value3.split())
    #     list4.append(value4.split())
    #     list5.append(value5.split())
    #     list6.append(value6.split())
    #     list7.append(value7.split())
    #     list8.append(value8)
    #     list9.append(value9.split())
    #     list10.append(value10)
    #
    #     # distance list
    #     sent_list = value1.split()
    #     e1_idx = sent_list.index("drug_1")
    #     e2_idx = sent_list.index("drug_2")
    #     dis1_list = []
    #     dis2_list = []
    #     for i in range(len(sent_list)):
    #         dis1_list.append(str(i-e1_idx))
    #         dis2_list.append(str(i-e2_idx))
    #     list11.append(dis1_list)
    #     list12.append(dis2_list)
    #
    # all_lists = (list1, list2, list3, list4, list5, list6, list7, list8, list9, list10, list11, list12)


    # ----------------------------------------------------



        value1 = lines[0]
        value2 = lines[1]
        value3 = lines[2]
        value4 = lines[3]
        value5 = lines[4]
        value6 = lines[5]

        list1.append(value1.split())
        list2.append(value2.split())
        list3.append(value3)

        # distance list
        sent_blind_list = value1.split()
        e1_idx = sent_blind_list.index("DRUG_1")
        e2_idx = sent_blind_list.index("DRUG_2")
        dis1_list = []
        dis2_list = []
        for i in range(len(sent_blind_list)):
            dis1_list.append(str(i-e1_idx))
            dis2_list.append(str(i-e2_idx))
        list4.append(dis1_list)
        list5.append(dis2_list)

        list6.append(value4)
        list7.append(value5.split())
        list8.append(value6)

    all_lists = (list1, list2, list3, list4, list5, list6, list7, list8)

    return all_lists



def dataPickle(tr_data_path, te_data_path, pickle_data_path):
    train_set = dataRead(tr_data_path)
    test_set = dataRead(te_data_path)

    with open(pickle_data_path, "wb") as handle:
        pkl.dump(train_set, handle)
        pkl.dump(test_set, handle)

    print ("save in ", pickle_data_path)


# execute
print ("pickle data ... ")
# dataPickle("./ddi_corpus/shortestpath/train_data.txt", "./ddi_corpus/shortestpath/test_data.txt", "./ddi_corpus/ddi_corpus.pickle")
# dataPickle("./ddi_corpus/dependencydfs/train_data.txt", "./ddi_corpus/dependencydfs/test_data.txt", "./ddi_corpus/ddi_corpus.pickle")
# dataPickle("./ddi_corpus/05negativefilt_/train_data.txt", "./ddi_corpus/05negativefilt/test_data.txt", "./ddi_corpus/ddi_corpus.pickle")
dataPickle("./ddi_corpus/05negativefilt_no_dependency/train_data.txt", "./ddi_corpus/05negativefilt_no_dependency/test_data.txt", "./ddi_corpus/ddi_corpus_wpd.pickle")