import os
from sklearn.model_selection import train_test_split
import random

#This code lets the user select which trajectories will be included in the test and train split.

root = '/home/usuaris/imatge/marc.bo/proj/headpose/mixed_depth/'

files = [f for f in os.listdir(root) if f != "splits" ]

r = []
for file1 in files:
    if file1 != "split.py" or file1 != "splits":
        r.append(file1)
random.shuffle(r)

train, test = train_test_split(r)
print(train)
dataTrain = []
dataTest = []
for file2 in train:
    for line in open(root + file2, 'r'):
        dataTrain.append(line)

for file3 in test:
    for line in open(root + file3, 'r'):
        dataTest.append(line)

with open(root+ 'splits/test_2_rgb.txt', 'a') as t:
        for item in dataTest:
            t.write(item)
with open(root + 'splits/train_2_rgb.txt', 'a') as p:
        for item in dataTrain:
            p.write(item)