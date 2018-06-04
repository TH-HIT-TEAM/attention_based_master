import copy
import re


def preprocess(r_path, w_path):
    print ("====================================")
    print ("process ", r_path, "...\n")
    data = open(r_path, "r")
    fw = open(w_path, "w")
    for s in data.read().strip().split("\n\n"):
        sent_id = s.strip().split("\n")[0].split("\t")[0]
        sent = s.strip().split("\n")[0].split("\t")[1]
        pair = s.strip().split("\n")[1:]

        # build entity dictionary & pair dictionary
        i = 0
        j = 0
        entity_dict = {}
        pair_dict = {}
        for p in pair:
            # print pair
            e1, e1_type, e1_span, e2, e2_type, e2_span, relation = p.strip().split("\t")
            entity_one = e1 + "@@" + e1_type + "@@" + e1_span
            entity_two = e2 + "@@" + e2_type + "@@" + e2_span

            # here e_value is for key, e_key is for value in entity dictionary
            if entity_one not in entity_dict:
                e_key = "E" + str(i)
                e_value = entity_one
                entity_dict[e_value] = e_key
                i += 1
            if entity_two not in entity_dict:
                e_key = "E" + str(i)
                e_value = entity_two
                entity_dict[e_value] = e_key
                i += 1

            # pair dictionary
            p_key = "P" + str(j)
            entity_one_id = entity_dict.get(entity_one)
            entity_two_id = entity_dict.get(entity_two)
            p_value = entity_one_id + "@@" + entity_two_id + "@@" + relation
            pair_dict[p_key] = p_value
            j += 1

        # transpose entity dictionary
        transpose_entity_dict = {v: k for k, v in entity_dict.items()}
        global_transpose_entity_dict = copy.copy(transpose_entity_dict)
        # print "-----", global_transpose_entity_dict

        # print "\noriginal sent: ", sent , "\n"
        new_sent = sent
        for key, value in transpose_entity_dict.items():
            entity, entity_type, entity_span = global_transpose_entity_dict.get(key).split("@@")
            # print entity, " ", entity_type, " ", entity_span
            if entity_span.find(";") > -1:
                # only consider two spans
                span1, span2 = entity_span.split(";")
                span1_start, span1_end = span1.split("-")
                span1_start = int(span1_start)
                span1_end = int(span1_end)
                span2_start, span2_end = span2.split("-")
                span2_start = int(span2_start)
                span2_end = int(span2_end)
                new_sent = new_sent[:span1_end + 1] + " " + new_sent[span2_start:span2_end + 1] + new_sent[
                                                                                                  span1_end + 1:]
                new_entity_span = str(span1_start) + "-" + str((span1_end + (span2_end - span2_start + 2)))
                new_entity = new_sent[span1_start:(span1_end + (span2_end - span2_start + 3))]
                # print "===blind sent: ", new_sent
                # print new_sent[span1_start:(span1_end + (span2_end - span2_start + 3))]


                # update the spans of other entities
                for en_key, en_value in global_transpose_entity_dict.items():
                    if en_key == key:
                        new_value = new_entity + "@@" + entity_type + "@@" + new_entity_span
                        global_transpose_entity_dict[key] = new_value
                        # print "....", global_transpose_entity_dict
                    else:
                        en, en_type, en_span = global_transpose_entity_dict.get(en_key).split("@@")
                        if en_span.find(";") == -1:
                            en_span_start, en_span_end = en_span.split("-")
                            en_span_start = int(en_span_start)
                            en_span_end = int(en_span_end)
                            if en_span_end <= span1_end:
                                en_span_start = en_span_start
                                en_span_end = en_span_end
                            elif en_span_start > span1_end:
                                en_span_start = en_span_start + (span2_end - span2_start + 2)
                                en_span_end = en_span_end + (span2_end - span2_start + 2)
                            else:
                                print ("Abnormal one!")
                                print (global_transpose_entity_dict.get(en_key))
                            new_en_span = str(en_span_start) + "-" + str(en_span_end)
                            new_en_value = en + "@@" + en_type + "@@" + new_en_span
                            global_transpose_entity_dict[en_key] = new_en_value
                            # print "++++", global_transpose_entity_dict
                        else:
                            en_span1, en_span2 = en_span.split(";")
                            en_span1_start, en_span1_end = en_span1.split("-")
                            en_span1_start = int(en_span1_start)
                            en_span1_end = int(en_span1_end)
                            if en_span1_end <= span1_end:
                                en_span1_start = en_span1_start
                                en_span1_end = en_span1_end
                            elif en_span1_start > span1_end:
                                en_span1_start = en_span1_start + (span2_end - span2_start + 2)
                                en_span1_end = en_span1_end + (span2_end - span2_start + 2)
                            else:
                                print ("Abnormal two!")
                                print (global_transpose_entity_dict.get(en_key))

                            en_span2_start, en_span2_end = en_span2.split("-")
                            en_span2_start = int(en_span2_start)
                            en_span2_end = int(en_span2_end)
                            if en_span2_end <= span1_end:
                                en_span2_start = en_span2_start
                                en_span2_end = en_span2_end
                            elif en_span2_start > span1_end:
                                en_span2_start = en_span2_start + (span2_end - span2_start + 2)
                                en_span2_end = en_span2_end + (span2_end - span2_start + 2)
                            else:
                                print ("Abnormal three!")

                            new_en_span = str(en_span1_start) + "-" + str(en_span1_end) + ";" + str(
                                en_span2_start) + "-" + str(en_span2_end)
                            new_en_value = en + "@@" + en_type + "@@" + new_en_span
                            global_transpose_entity_dict[en_key] = new_en_value
                            # print "====", global_transpose_entity_dict

        # print "----------------------------------"
        # print "\\\\\\\\\\\\\n"

        print (sent_id)
        print (new_sent)
        fw.write(sent_id + "\t" + new_sent + "\n")
        for p_key, p_value in pair_dict.items():
            entity_one_id, entity_two_id, relation = p_value.split("@@")
            entity_one = global_transpose_entity_dict.get(entity_one_id)
            entity_two = global_transpose_entity_dict.get(entity_two_id)
            e1, e1_type, e1_span = entity_one.split("@@")
            e2, e2_type, e2_span = entity_two.split("@@")

            pair_line = e1 + "\t" + e1_type + "\t" + e1_span + "\t" + e2 + "\t" + e2_type + "\t" + e2_span + "\t" + relation
            fw.write(pair_line + "\n")
            print (pair_line)
        fw.write("\n")
        print ("-------------------------------------")


# execute
preprocess("./ddi_corpus/01processxml/train_data.txt", "./ddi_corpus/02entitycorrect/train_data.txt")
preprocess("./ddi_corpus/01processxml/test_data.txt", "./ddi_corpus/02entitycorrect/test_data.txt")
