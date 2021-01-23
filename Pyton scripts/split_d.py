import os
import sklearn
from sklearn.model_selection import train_test_split

#This code lets the user select which trajectories will be included in the test and train split.

root = '/mnt/gpid07/imatge/marc.bo/proj/headpose/paths_d'

files = [f for f in os.listdir(root) if os.path.isfile(f) and f != 'split.py']

r = []
for file1 in files:
    if file1 != "split2.py":
        r.append(file1)

train, test = train_test_split(r)

dataTrain = []
dataTest = []
for file2 in train:
    for line in open(file2, 'r'):
        dataTrain.append(line)

for file3 in test:
    for line in open(file3, 'r'):
        dataTest.append(line)

with open('splits/test_2_depth.txt', 'a') as t:
        for item in dataTest:
            t.write(item)
with open('splits/train_2_depth.txt', 'a') as p:
        for item in dataTrain:
            p.write(item)