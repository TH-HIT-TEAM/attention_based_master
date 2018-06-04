import os
import shutil
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import itertools


def resetDir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
        os.makedirs(dir)
    else:
        os.makedirs(dir)



att_image_dir_correct = "./att_image/correct"
resetDir(att_image_dir_correct)
att_image_dir_wrong = "./att_image/wrong"
resetDir(att_image_dir_wrong)

f = open("../best_result/test_result.txt", "r")
results = f.read().strip().split("\n\n")
count = 0

for result in results:

    count += 1
    if count > 100:
        continue

    att_weight_list = []
    sent_list = []

    terms = result.strip().split("\n")
    id = terms[0]
    sent = terms[3]
    label = terms[7]
    pred = terms[8]
    att_weight = terms[10]
    print(count, "\t", id)

    sent_list.append([w for w in sent.split()])
    text = np.array(sent_list)
    att_weight_list.append([float(a) for a in att_weight.split()])
    data = np.array(att_weight_list)


    plt.figure(figsize=(float(len(att_weight_list[0])) * 1.5, float(len(att_weight_list[0])) * 1.5 / 20))
    plt.yticks(np.arange(0, 2), ["prediction: " + pred, "label: " + label])
    # plt.axis('off')


    # draw text
    for i, j in itertools.product(range(text.shape[0]), range(text.shape[1])):
        plt.text(j + 0.5, i + 0.5, text[i, j],
                 horizontalalignment="center",
                 color="black",
                 fontsize=10)

    # draw att
    plt.pcolor(data, cmap=matplotlib.cm.Blues, norm=matplotlib.colors.Normalize(vmin=np.min(data), vmax=np.max(data)))

    # plt.colorbar()
    # plt.tight_layout()
    # plt.show()
    if pred==label:
        plt.savefig(att_image_dir_correct + "/" + id + ".png")
    else:
        plt.savefig(att_image_dir_wrong + "/" + id + ".png")
    plt.close()


print("done")






