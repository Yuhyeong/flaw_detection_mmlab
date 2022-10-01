import os
import pickle
from pprint import pprint

f = open("../datasets/train.pkl", 'rb')
data = pickle.load(f)

wrong_list = []

for ann in data:
    labels = ann['ann']['bboxes']

    # if ann['filename'] == '046_58.jpg':
    #     pprint(labels)

    for line in labels:
        xmin, ymin, xmax, ymax = line[0], line[1], line[2], line[3]
        if xmin > xmax or ymin > ymax:
            wrong_list.append(ann['filename'])

pprint(wrong_list)

