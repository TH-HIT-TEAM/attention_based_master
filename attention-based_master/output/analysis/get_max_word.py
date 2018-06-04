import shutil
import os
from nltk.stem.lancaster import LancasterStemmer

def resetDir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
        os.makedirs(dir)
    else:
        os.makedirs(dir)



max_word_dir = "./max_word"
resetDir(max_word_dir)

f = open("../best_result/test_result.txt", "r")
results = f.read().strip().split("\n\n")

lancaster_stemmer = LancasterStemmer()
max_word_dict = {}
max_pos_dict = {}
for result in results:
    terms = result.strip().split("\n")
    id = terms[0]
    sent = terms[3]
    pos = terms[4]
    att_weight = terms[10]

    att_weight_list = [float(a) for a in att_weight.split()]
    temp = att_weight_list
    max_index_list = []
    for i in range(3):
        max_index = temp.index(max(temp))
        max_index_list.append(max_index)
        temp[max_index] = 0
    # print (max_index_list)

    for max_index in max_index_list:
        word_list = [w for w in sent.split()]
        # max_word = lancaster_stemmer.stem(word_list[max_index])
        max_word = word_list[max_index]
        if max_word in max_word_dict:
            max_word_dict[max_word] += 1
        else:
            max_word_dict[max_word] = 0

        pos_list = [p for p in pos.split()]
        max_pos = pos_list[max_index]
        if max_pos in max_pos_dict:
            max_pos_dict[max_pos] += 1
        else:
            max_pos_dict[max_pos] = 0



max_word_dict = sorted(max_word_dict.items(), key = lambda x:x[1],reverse = True)
print (max_word_dict)

max_pos_dict = sorted(max_pos_dict.items(), key = lambda x:x[1],reverse = True)
print (max_pos_dict)





