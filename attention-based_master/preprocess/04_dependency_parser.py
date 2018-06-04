import os
from nltk.parse import stanford
from nltk.tree import Tree
from nltk import word_tokenize
import networkx as nx
import re

java_path = "/usr/lib/jvm/jdk1.8.0_152/bin"
os.environ["JAVAHOME"] = java_path

os.environ["STANFORD_PARSER"] = "resource/stanford-parser.jar"
os.environ["STANFORD_MODELS"] = "resource/stanford-parser-3.8.0-models.jar"
dep_parser = stanford.StanfordDependencyParser(model_path="resource/englishPCFG.ser.gz", java_options="-mx1500m")
# parser = stanford.StanfordParser(model_path="resource/englishPCFG.ser.gz", java_options="-mx1500m")


def list_nodes_depth(root, list):
    if isinstance(root, Tree):
        list.append(root.label())
        for child in root:
            list_nodes_depth(child, list)
    else:
        list.append(root)
    return list


def get_depth_list(tree):
    list = []
    list = list_nodes_depth(tree, list)
    line = ""
    for word in list:
        line = line + word + " "
    return line.strip()


def list_nodes_breadth(root):
    result = []
    if isinstance(root, Tree):
        result.append(root.label())
    else:
        result.append(root)

    queue = []
    queue.append(root)
    while len(queue) > 0:
        node = queue.pop(0)
        for child in node:
            if isinstance(child, Tree):
                result.append(child.label())
                queue.append(child)
            else:
                result.append(child)
    return result


def get_breadth_list(tree):
    list = list_nodes_breadth(tree)
    line = ""
    for word in list:
        line = line + word + " "
    return line.strip()


def raw_parse_dependency(sent):
    for parse in dep_parser.raw_parse(sent):
        return parse


# def raw_parse_phrase(sent):
#     for parse in parser.raw_parse(sent):
#         return parse


def preprocess(r_path, w_path):
    data = open(r_path, "r")
    fw = open(w_path, "w")

    print ("processing: ", r_path)
    count = 0
    for s in data.read().strip().split("\n\n"):
        sent_blind = s.strip().split("\n")[0]
        relation = s.strip().split("\n")[1]
        sent_id = s.strip().split("\n")[2]
        sent = s.strip().split("\n")[3]
        entities = s.strip().split("\n")[4]

        parse = raw_parse_dependency(sent_blind)

        # shorest dependency path ============================
        conll = parse.to_conll(10)  # 3,4,10
        # 1. ID: Word index, integer starting at 1 for each new sentence; may be a range for tokens with multiple words.
        # 2. FORM: Word form or punctuation symbol.
        # 3. LEMMA: Lemma or stem of word form.
        # 4. CPOSTAG: Universal part-of-speech tag drawn from our revised version of the Google universal POS tags.
        # 5. POSTAG: Language-specific part-of-speech tag; underscore if not available.
        # 6. FEATS: List of morphological features from the universal feature inventory or from a defined language-specific extension; underscore if not available.
        # 7. HEAD: Head of the current token, which is either a value of ID or zero (0).
        # 8. DEPREL: Universal Stanford dependency relation to the HEAD (root iff HEAD = 0) or a defined language-specific subtype of one.
        # 9. DEPS: List of secondary dependencies (head-deprel pairs).
        # 10. MISC: Any other annotation.
        deps = [dep for dep in conll.strip().split("\n")]

        word_dict = {}
        word_dict["0"] = "0@@EMPTY_ROOT@@none"
        for dep in deps:
            terms = dep.split("\t")
            id = terms[0]
            form = terms[1]
            postag = terms[4]
            deprel = terms[7]
            word_dict[id] = id + "@@" + form + "@@" + postag + "@@" + deprel

            if form == "DRUG_1":
                source = word_dict[id]
            if form == "DRUG_2":
                target = word_dict[id]

        edges_list = []
        for dep in deps:
            terms = dep.split("\t")
            id = terms[0]
            head = terms[6]

            govern = word_dict.get(head)
            depend = word_dict.get(id)
            tuple = (govern, depend)
            edges_list.append(tuple)

        # print edges_list
        graph = nx.Graph(edges_list)  # see https://networkx.github.io/documentation/networkx-1.9/tutorial/tutorial.html
        shortest_path_list = nx.shortest_path(graph, source=source, target=target)
        # print shortest_path_list

        word_path_string = ""
        pos_path_string = ""
        deprel_path_string = ""
        for i in range(len(shortest_path_list)):
            terms = shortest_path_list[i].split("@@")
            word = terms[1]
            pos = terms[2]
            deprel = terms[3]

            word_path_string = word_path_string + " " + word
            pos_path_string = pos_path_string + " " + pos

            if i < (len(shortest_path_list) - 1):
                tuple = (shortest_path_list[i], shortest_path_list[i + 1])
                tuple_transpose = (shortest_path_list[i + 1], shortest_path_list[i])
                if tuple in edges_list :
                    deprel = shortest_path_list[i + 1].split("@@")[3]
                    deprel = deprel + "_r"
                elif tuple_transpose in edges_list:
                    deprel = deprel + "_l"
            else:
                deprel = "end_tag"
            deprel_path_string = deprel_path_string + " " + deprel

        # print word_path_string.strip()
        # print pos_path_string.strip()
        # print deprel_path_string.strip()
        # parse.tree().draw()

        # tree_line ===================================
        # add _tree_pos() and tree_pos() function in /home/weby/anaconda2/envs/tensorflow0.12_py27/lib/python2.7/site-packages/nltk/parse/dependencygraph.py

        # def _tree_pos(self, i):
        #     """ Turn dependency graphs into NLTK trees.
        #
        #     :param int i: index of a node
        #     :return: either a word (if the indexed node is a leaf) or a ``Tree``.
        #     """
        #     node = self.get_by_address(i)
        #     word = node['tag']
        #     deps = sorted(chain.from_iterable(node['deps'].values()))
        #
        #     if deps:
        #         return Tree(word, [self._tree_pos(dep) for dep in deps])
        #     else:
        #         return word

        # def tree_pos(self):
        #     """
        #     Starting with the ``root`` node, build a dependency tree using the NLTK
        #     ``Tree`` constructor. Dependency labels are omitted.
        #     """
        #     node = self.root
        #
        #     word = node['tag']
        #     deps = sorted(chain.from_iterable(node['deps'].values()))
        #     return Tree(word, [self._tree_pos(dep) for dep in deps])


        tree = parse.tree()
        tree_pos = parse.tree_pos()
        tree_line = "(" + tree.label()
        tree_pos_line = "(" + tree_pos.label()
        for i in range(len(tree)):
            tree_line = tree_line + " " + str(tree[i])
            tree_pos_line = tree_pos_line + " " + str(tree_pos[i])
        tree_line = " ".join(word_tokenize(tree_line + ")"))
        tree_pos_line = " ".join(word_tokenize(tree_pos_line + ")"))
        tree_line = tree_line + ")"
        tree_line = re.sub("\(", " ( ", tree_line)
        tree_line = re.sub("\)", " ) ", tree_line)

        tree_pos_line = tree_pos_line + ")"
        tree_pos_line = re.sub("\(", " ( ", tree_pos_line)
        tree_pos_line = re.sub("\)", " ) ", tree_pos_line)

        # print tree_line
        # print "-------------------"

        fw.write(sent_blind + "\n")
        fw.write(relation + "\n")
        fw.write(word_path_string.strip() + "\n")
        fw.write(deprel_path_string.strip() + "\n")
        fw.write(pos_path_string.strip() + "\n")
        fw.write(tree_line.strip() + "\n")
        fw.write(tree_pos_line.strip() + "\n")
        fw.write(sent_id + "\n")
        fw.write(sent + "\n")
        fw.write(entities + "\n")
        fw.write("\n")

        count += 1
        print ("."),
        if count % 100 == 0:
            print (count)


# execute
preprocess("./ddi_corpus/03pairwithsent/train_data.txt", "./ddi_corpus/04dependencydfs/train_data.txt")
preprocess("./ddi_corpus/03pairwithsent/test_data.txt", "./ddi_corpus/04dependencydfs/test_data.txt")










# test code
# ===================================================================================================

#
# print "---------------------------"
# tree = parse.tree()
# print tree, "\n"
# draw dependency tree
# tree.draw()


# print "-------------------"
# sent = "The role of p27 ( Kip1 ) in DRUG_1 -enhanced DRUG_2 cytotoxicity in human ovarian cancer cells ."
# parse = raw_parse_dependency(sent)
#
#
# tree = parse.tree()
# tree_line = "(" + tree.label()
# for node in list(tree):
#     tree_line = tree_line + " " + str(node)
# tree_line = " ".join(word_tokenize(tree_line + ")"))
# print "tree_line: ", tree_line, "\n"
#
# conll = parse.to_conll(10)
# print conll
#
#
# parse = raw_parse_phrase(sent)
# parse.draw()



# print tree.label()
# for node in tree:
#     print node

#
# triples = parse.triples()
# print list(triples)
