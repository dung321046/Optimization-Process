import pandas as pd
from collections import Counter
from scipy.stats import entropy


def cal_entropy(counter, N):
    prob_dis = []
    for ele in counter:
        prob_dis.append(counter[ele] / N)
    # print(prob_dis)
    # print(entropy(np.asarray(prob_dis)))
    return entropy(prob_dis, base=2)


data = pd.read_csv("train_data.txt")
data = data.to_numpy()
n = len(data)
ents = []
len_atts = []
for c in range(1, len(data[0])):
    counter = Counter(data[:, c])
    ent = cal_entropy(counter, n)
    len_atts.append(len(counter))
    ents.append(ent)
from visualization import histogram

histogram.histogram_uni(ents, 20, "Entropy of attributes", "Entropy")
histogram.histogram_uni(len_atts, 20, "Number of unique values in each attribute", "N-of-Unique-values")
# ents = sorted(ents)
# print(ents)
# print(ents[:30])
# print(len(ents[:30]))
minat = 100
for i, ent in enumerate(ents):
    if ent < 0.01:
        print(i, " <> ", len_atts[i], end=",")
        minat = min(i, minat)
print()
print("Min att", minat)
for i, ent in enumerate(ents):
    if ent > 4:
        print(i, " <> ", len_atts[i], end=",")
print()
for i, len_att in enumerate(len_atts):
    if len_att < 3:
        print(i, " <> ", len_atts[i], end=",")
