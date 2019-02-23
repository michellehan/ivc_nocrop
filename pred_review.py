from __future__ import division
import sys, os
import numpy as np
from sklearn.metrics import f1_score

data_dir = sys.argv[1]

def printwrite(outfile, text):
    outfile.write('%s\n' %text)
    print(text)
    

def unpack(data_dir):
    outname = data_dir.replace('npz', 'txt')
    outfile = open(outname, 'w')

    printwrite(outfile, data_dir)
    data = np.load(data_dir)

    targets = data['target']
    pred = data['pred']
    pred_ema = data['ema_pred']

    classes, counts = np.unique(targets.argmax(axis=1), return_counts=True)
    totals = dict(zip(classes, counts)).values()

    corrects = [0] * len(totals)
    corrects_ema = [0] * len(totals)
    printwrite(outfile, '* = incorrect prediction by regular model')
    printwrite(outfile, '** = incorrect prediction by EMA model')
    printwrite(outfile, '*** = incorrect prediction by both models')
    for i in range(targets.shape[0]):    
        t = targets[i].argmax(axis=0)
        p = pred[i].argmax(axis=0)
        pe = pred_ema[i].argmax(axis=0)

        if int(t==p) == 1: corrects[t] += 1
        if int(t==pe) == 1: corrects_ema[t] +=1

        if t==p and t==pe:
            printwrite(outfile, '[%s]\ttarget: %s \tpred: %s \tpred_ema: %s' %(i, t, p, pe))
        elif not t==p:
            if t==pe: 
                printwrite(outfile, '[%s]\ttarget: %s \tpred: %s \tpred_ema: %s *' %(i, t, p, pe))
            if not t==pe:
                printwrite(outfile, '[%s]\ttarget: %s \tpred: %s \tpred_ema: %s ***' %(i, t, p, pe))
        elif not t==pe: 
            printwrite(outfile, '[%s]\ttarget: %s \tpred: %s \tpred_ema: %s **' %(i, t, p, pe))


    f1 = f1_score(y_true=targets, y_pred=pred, average='weighted') * 100
    f1_list = list(f1_score(y_true=targets, y_pred=pred, average=None))
    f1_list = [round(i * 100,2) for i in f1_list]

    ema_f1 = f1_score(y_true=targets, y_pred=pred_ema, average='weighted') * 100
    ema_f1_list = list(f1_score(y_true=targets, y_pred=pred_ema, average=None)) 
    ema_f1_list = [round(i * 100,2) for i in ema_f1_list]
    
    printwrite(outfile, '\nF1 overall: %s \nF1 by class: %s' %(f1, f1_list))
    printwrite(outfile, '\nF1 EMA overall: %s \nF1 EMA by class: %s' %(ema_f1, ema_f1_list))

    acc = [round(c/t*100, 2) for c,t in zip(corrects, totals)]
    ema_acc = [round(c/t*100, 2) for c,t in zip(corrects_ema, totals)]
    printwrite(outfile, '\nAcc overall: %s \nAcc by class: %s' %(data['auc']*100, acc))
    printwrite(outfile, '\nAcc EMA overall: %s \nAcc EMA by class: %s' %(data['ema_auc']*100, ema_acc))
    #print('\nAcc: %s \t Acc EMA: %s' %(data['auc'], data['ema_auc']))
    

#data_dir = "/home/mihan/projects/ivc_nocrop/test_pred/full_cls14/test_full_resnet50_b32_label32_primary.npz"
unpack(data_dir)

#data_dir = "/home/mihan/ivc_kfold/test_pred/full_cls11/test_full_resnet50_b64_label64_primary.npz"
#unpack(data_dir)
