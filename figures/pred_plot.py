from __future__ import division
import sys, os

import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

#data_dir = sys.argv[1]
data_dir = "/home/mihan/projects/ivc/lr0.1decay25momentum0.8/test_pred/full_cls14/test_full_resnet50_b32_label32_primary.npz"

title='Confusion Matrix'
cmap = plt.cm.Blues
normalize = True

data = np.load(data_dir)

targets_data = data['target']
pred_data = data['pred']

classes, counts = np.unique(targets_data.argmax(axis=1), return_counts=True)
totals = dict(zip(classes, counts)).values()
labels = np.asarray(["ALN", "BD", "BE", "BG", "BM", "BSN", "C", "CP", "G", "GT", "Os", "On", "Tr", "Tu"])

targets = []
preds = []
for i in range(targets_data.shape[0]):    
    t = targets_data[i].argmax(axis=0)
    p = pred_data[i].argmax(axis=0)
    targets.append(t)
    preds.append(p)

print(targets)
print(preds)



cm = confusion_matrix(targets, preds)
np.set_printoptions(precision = 2)

if normalize: cm = cm.astype('float') / cm.sum(axis = 1)[:,np.newaxis]
print(cm)

plt.figure()
plt.imshow(cm, interpolation = 'nearest', cmap=cmap)
plt.title(title)
plt.colorbar()
tick_marks = np.arange(len(classes))
#tick_marks = labels
plt.xticks(tick_marks, labels, rotation = 45)
plt.yticks(tick_marks, labels)

fmt = '.2f' if normalize else 'd'
thresh = cm.max() / 2.

for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout() 

plt.show()

