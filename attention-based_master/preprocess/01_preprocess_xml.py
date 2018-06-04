import os
# import sys
# reload(sys)
# sys.setdefaultencoding("utf-8")
from xml.dom.minidom import parse
import xml.dom.minidom


# readXML ================================================================
def readXML(dir_list):
    data = []
    for d in dir_list:
        file_list = os.listdir(d)
        for fname in file_list:
            DOMTree = xml.dom.minidom.parse(d + "/" + fname)
            Data = DOMTree.documentElement
            sents = Data.getElementsByTagName("sentence")
            for sent in sents:
                sent_id = sent.getAttribute("id")
                sent_text = sent.getAttribute("text").strip()
                ent_dict = {}
                pair_list = []

                entities = sent.getElementsByTagName("entity")
                for entity in entities:
                    d_type = entity.getAttribute("type")
                    d_id = entity.getAttribute("id")
                    d_ch_of = entity.getAttribute("charOffset")
                    d_text = entity.getAttribute("text")
                    ent_dict[d_id] = [d_text, d_type, d_ch_of]

                pairs = sent.getElementsByTagName("pair")
                for pair in pairs:
                    p_id = pair.getAttribute("id")
                    e1 = pair.getAttribute("e1")
                    entity1 = ent_dict[e1]
                    e2 = pair.getAttribute("e2")
                    entity2 = ent_dict[e2]
                    ddi = pair.getAttribute("ddi")
                    if ddi == "true":
                        if "type" in pair.attributes.keys():
                            ddi = pair.getAttribute("type")
                        else:
                            ddi = "int"
                    pair_list.append([entity1, entity2, ddi])

                data.append([sent_id, sent_text, pair_list, fname])
    return data


# rewriteData ================================================================
def rewriteData(path, data):
    i = 0
    fw = open(path, "w")
    for sid, stext, pair, file in data:
        if len(pair) == 0:
            continue;
        fw.write(file+ "@@" + sid + "\t" + stext + "\n")
        for e1, e2, ddi in pair:
            fw.write(e1[0] + "\t" + e1[1] + "\t" + e1[2] + "\t" + e2[0] + "\t" + e2[1] + "\t" + e2[2] + "\t" + ddi)
            fw.write("\n")
            i = i + 1
        fw.write("\n")
    print (path, ": ", i)


# preprocess ================================================================
def preprocess():
    tr_med = "./ddi_corpus/00sourcexml/Train/MedLine"
    tr_drug = "./ddi_corpus/00sourcexml/Train/DrugBank"
    te_med = "./ddi_corpus/00sourcexml/Test/MedLine"
    te_drug = "./ddi_corpus/00sourcexml/Test/DrugBank"

    tr_data = readXML([tr_med, tr_drug])
    te_data = readXML([te_med, te_drug])

    rewriteData("ddi_corpus/01processxml/train_data.txt", tr_data)
    rewriteData("ddi_corpus/01processxml/test_data.txt", te_data)


# execute ================================================================
preprocess()
