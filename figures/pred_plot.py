from __future__ import division
import sys, os

import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

#data_dir = sys.argv[1]
data_dir = "/home/mihan/projects/ivc/ivc_nocrop/test_pred/4_0.00_cls14/test_4_resnet50_b32_label32_primary.npz"

##title='Confusion Matrix'
cmap = plt.cm.Blues
normalize = True

data = np.load(data_dir)

targets_data = data['target']
pred_data = data['pred']

classes, counts = np.unique(targets_data.argmax(axis=1), return_counts=True)
totals = dict(zip(classes, counts)).values()
labels = np.asarray(["ALN", "De", "EX", "G2", "Me", "SN", "Ce", "CP", "GS", "GT", "Os", "On", "Tr", "Tu"])
labels = np.asarray(["ALN", "G2", "G2X", "Mer", "Den", "SN", "On", "Cel", "CelP", "GT", "GFSS", "GFTi", "OptE", "TrpE"])

targets = []
preds = []
for i in range(targets_data.shape[0]):    
    t = targets_data[i].argmax(axis=0)
    p = pred_data[i].argmax(axis=0)
    targets.append(t)
    preds.append(p)

print(targets)
print(preds)

targets = [0, 0, 0, 0, 0, 0, 0, 0, 4,4,4,4,4,4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1,1,1,1,1, 3,3,3,3,3,3, 5, 5, 5, 5, 5, 5, 5, 7,7,7,7,7, 8,8,8,8,8,8,8,8,8, 10,10,10,10,10,10,10,10,10,10,10,10,10,10, 11,11,11,11,11,11,11,11,11,11,11, 12,12,12,12,12,12,12,12,12,12,12,12,12,12, 6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6, 13,13,13,13,13,13,13,13,13,13,13, 9,9,9,9,9,9,9,9,9,9,9]

preds = [0, 0, 0, 0, 0, 0, 0, 0, 4,4,4,4,4,4, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1,1,1,1,1, 3,3,3,3,3,2, 5, 5, 5, 5, 5, 5, 5, 7,7,7,7,7, 8,8,8,8,8,8,8,8,8, 10,10,10,10,10,10,10,10,10,10,10,10,10,10, 11,11,11,11,11,11,11,11,11,11,11, 12,12,12,12,12,12,13,12,13,12,12,12,12,12, 6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6, 13,13,13,13,13,13,13,13,13,13,13, 9,9,9,9,9,9,9,9,9,9,9]



cm = confusion_matrix(targets, preds)
np.set_printoptions(precision = 2)

if normalize: cm = cm.astype('float') / cm.sum(axis = 1)[:,np.newaxis]
print(cm)

plt.figure()
plt.figure(figsize=(10,10))
plt.rcParams.update({'font.size': 12})
plt.imshow(cm, interpolation = 'nearest', cmap=cmap)
#plt.title(title)
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

#hfont = {'fontname':'Times New Roman'}

#plt.ylabel('True Filter Type')
#plt.xlabel('Model-Predicted Filter Type')
plt.tight_layout() 

plt.savefig('confusionMatrix.png', format='png', dpi=1200)
plt.savefig('confusionMatrix.svg', format='svg', dpi=1200)

#plt.show()

