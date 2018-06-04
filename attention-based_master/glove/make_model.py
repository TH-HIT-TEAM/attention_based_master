# add word count & dimension in the first line
import gensim
import os
import shutil
import hashlib
from sys import platform


# word count
def getFileLineNums(filename):
    f = open(filename, 'r')
    count = 0
    for line in f:
        count += 1
    return count


# add a line
def prepend_line(infile, outfile, line):
    with open(infile, 'r') as old:
        with open(outfile, 'w') as new:
            new.write(str(line) + "\n")
            shutil.copyfileobj(old, new)


def prepend_slow(infile, outfile, line):
    with open(infile, 'r') as fin:
        with open(outfile, 'w') as fout:
            fout.write(line + "\n")
            for line in fin:
                fout.write(line)

def make_model(filename, dimension):
    num_lines = getFileLineNums(filename)
    gensim_file = 'glove_model.txt'
    gensim_first_line = "{} {}".format(num_lines, dimension)
    # Prepends the line.
    if platform == "linux" or platform == "linux2":
        prepend_line(filename, gensim_file, gensim_first_line)
    else:
        prepend_slow(filename, gensim_file, gensim_first_line)

    print "finish making glove model file"


make_model("glove.6B.100d.txt", 100)

# def load_glove(filename):
#     num_lines = getFileLineNums(filename)
#     gensim_file = 'glove_model.txt'
#     gensim_first_line = "{} {}".format(num_lines, 300)
#     # Prepends the line.
#     if platform == "linux" or platform == "linux2":
#         prepend_line(filename, gensim_file, gensim_first_line)
#     else:
#         prepend_slow(filename, gensim_file, gensim_first_line)
#
#     model = gensim.models.KeyedVectors.load_word2vec_format(gensim_file)
#
#     return model
#
#
# glove_model = load_glove('glove.6B.300d.txt')
# if "in" in glove_model:
#     print "yes"
# else:
#     print "no"
#
# if "iiiii" in glove_model:
#     print "kk"
# else:
#     print "ooo"