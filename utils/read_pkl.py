import os
import pickle
from pprint import pprint

f = open("../datasets/train.pkl", 'rb')
data = pickle.load(f)
pprint(data)



img_list = os.listdir("../datasets/train/data")

print(len(data))

print(len(img_list))
