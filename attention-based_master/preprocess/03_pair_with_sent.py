import copy
import re
import nltk
from nltk import word_tokenize


def washData(sent):
    sent = " ".join(word_tokenize(sent))
    sent = re.sub("[^\s]*\DRUG_1[^\s]*", "DRUG_1", sent)  # DRUG_1/6
    sent = re.sub("[^\s]*\DRUG_2[^\s]*", "DRUG_2", sent)  # DRUG_2/6
    sent = re.sub("[^\s]*\DRUG_N[^\s]*", "DRUG_N", sent)  # DRUG_N-fortified
    sent = re.sub("\s\.*\~*\/*\d+\,*\.*\d*(\s|$)", " TAG_OF_DIGIT ", sent)  # 100 or 1,200 or 3.9 or .014
    sent = re.sub("\s\.*\d+\,*\.*\d+\-+\d+\,*\.*\d+(\s|$)", " TAG_OF_DIGIT ", sent)  # 1.2-2.3
    sent = re.sub("\s\d+\/+\d+(\s|$)", " TAG_OF_RATIO ", sent)  # 2/3
    sent = re.sub("(\s|\DRUG_N|\;|\,)*\DRUG_1(\s|\DRUG_N|\;|\,)*\DRUG_2(\s|\DRUG_N|\;|\,)*", " DRUG_1 DRUG_2 ", sent)  # and DRUG_N DRUG_N DRUG_1 DRUG_N DRUG_N DRUG_N DRUG_2 DRUG_N ( presumptive Q-type )
    sent = re.sub("(\s|\DRUG_N|\;|\,)*\DRUG_2(\s|\DRUG_N|\;|\,)*\DRUG_1(\s|\DRUG_N|\;|\,)*", " DRUG_1 DRUG_2 ", sent)  # and DRUG_N DRUG_N DRUG_2 DRUG_N DRUG_N DRUG_N DRUG_1 DRUG_N ( presumptive Q-type )
    sent = re.sub("(\s|\DRUG_N|\;|\,)*\DRUG_1(\s|\DRUG_N|\;|\,)*", " DRUG_1 ", sent)  # and DRUG_N DRUG_N DRUG_N DRUG_1 DRUG_N ( presumptive Q-type )
    sent = re.sub("(\s|\DRUG_N|\;|\,)*\DRUG_2(\s|\DRUG_N|\;|\,)*", " DRUG_2 ", sent)  # and DRUG_N DRUG_N DRUG_N DRUG_2 DRUG_N ( presumptive Q-type )
    sent = re.sub("(\s|\DRUG_N|\;|\,)*\DRUG_N(\s|\DRUG_N|\;|\,)*", " DRUG_N ", sent)  # and DRUG_N DRUG_N DRUG_N DRUG_N DRUG_N ( presumptive Q-type )
    # sent = re.sub("\s\DRUG_1( and | or )+\DRUG_N", " DRUG_1", sent)  # DURG_1 and DRUG_N
    # sent = re.sub("\s\DRUG_2( and | or )+\DRUG_N", " DRUG_2", sent)  # DURG_2 and DRUG_N
    # sent = re.sub("\s\DRUG_N( and | or )+\DRUG_1", " DRUG_1", sent)  # DURG_N and DRUG_1
    # sent = re.sub("\s\DRUG_N( and | or )+\DRUG_1", " DRUG_2", sent)  # DURG_N and DRUG_2
    sent = re.sub("''", "", sent)
    sent = re.sub("``", "", sent)
    sent = re.sub("\"", "", sent)
    sent = re.sub("\.\.", ".", sent)
    # sent = re.sub("[+\.\!\/\-,:;@#$%^*()\-\"\']+|[+——！，：；。？、~@#￥%……&*（）]+", "", sent)
    # sent = re.sub("\s\s", " ", sent)
    return sent.strip()


def preprocess(r_path, w_path):
    data = open(r_path, "r")
    fw = open(w_path, "w")
    for s in data.read().strip().split("\n\n"):
        sent_id = s.strip().split("\n")[0].split("\t")[0]
        sent = s.strip().split("\n")[0].split("\t")[1]
        pair = s.strip().split("\n")[1:]

        span_list = []
        for p in pair:
            e1_span = p.strip().split("\t")[2]
            e2_span = p.strip().split("\t")[5]
            if e1_span not in span_list:
                span_list.append(e1_span)
            if e2_span not in span_list:
                span_list.append(e2_span)

        for p in pair:
            e1, e1_type, e1_span, e2, e2_type, e2_span, relation = p.strip().split("\t")
            # print sent
            # print p

            e1_start, e1_end = e1_span.split("-")
            e1_start = int(e1_start)
            e1_end = int(e1_end)
            e2_start, e2_end = e2_span.split("-")
            e2_start = int(e2_start)
            e2_end = int(e2_end)

            replace_e1 = 'a' * (e1_end - e1_start + 1)
            sent_blind = sent[:e1_start] + replace_e1 + sent[e1_end + 1:]
            replace_e2 = 'b' * (e2_end - e2_start + 1)
            sent_blind = sent_blind[:e2_start] + replace_e2 + sent_blind[e2_end + 1:]


            replace_en_list = []
            other_span_list = set(span_list) - set([e1_span, e2_span])
            for en_span in other_span_list:
                en_start, en_end = en_span.split("-")
                en_start = int(en_start)
                en_end = int(en_end)

                replace_en = 'c' * (en_end - en_start + 1)
                sent_blind = sent_blind[:en_start] + replace_en + sent_blind[en_end + 1:]
                replace_en_list.append(replace_en)


            # print sent_blind


            sent_blind = sent_blind.replace(replace_e1, ' DRUG_1 ')
            sent_blind = sent_blind.replace(replace_e2, ' DRUG_2 ')

            for replace_en in replace_en_list:
                sent_blind = sent_blind.replace(replace_en, 'DRUG_N')
            sent_blind = re.sub("DRUG_N\c*", " DRUG_N ", sent_blind)
            # print sent_blind


            sent_blind = washData(sent_blind)
            print (sent_blind)

            # contain DRUG_1 and DRUG_2 in the same time? if no error exception, YES!
            # index_d1 = sent_blind.index('DRUG_1')
            # index_d2 = sent_blind.index('DRUG_2')
            # print index_d1, " ", index_d2


            # part of speech
            sent_blind_pos = nltk.pos_tag(sent_blind.strip().split())
            sent_blind_pos_str = ""
            for pos in sent_blind_pos:
                sent_blind_pos_str = sent_blind_pos_str + " " + pos[1]
            sent_blind_pos_str = sent_blind_pos_str.strip()
            # print (len(sent_blind.split()))
            # print (len(sent_blind_pos_str.split()))
            # print ("-----")

            fw.write(sent_blind.strip() + "\n")
            fw.write(sent_blind_pos_str.strip() + "\n")
            fw.write(relation + "\n")
            fw.write(sent_id + "\n")
            fw.write(sent.strip() + "\n")
            fw.write(e1 + "\t" + e1_type + "\t" + e2 + "\t" + e2_type + "\n")
            fw.write("\n")


preprocess("./ddi_corpus/02entitycorrect/train_data.txt", "./ddi_corpus/03pairwithsent/train_data.txt")
preprocess("./ddi_corpus/02entitycorrect/test_data.txt", "./ddi_corpus/03pairwithsent/test_data.txt")
