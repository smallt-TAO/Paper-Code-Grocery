import sys
import os
import numpy as np
#thresholds = np.arange(0.0, 0.99, 0.005)
thresholds = np.arange(0.0, 0.99, 0.005)

for threshold in thresholds:
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    total = 0
    for line in open("distance.txt"):
        label = line.split("\t")[1]
        if label != 'unknow':
            total += 1
        predictor = float(line.split("\t")[2])
        if predictor > threshold and label == "YES":
            tp += 1
        if predictor < threshold and label == "NO":
            tn += 1
        if predictor > threshold and label == "NO":
            fp += 1
        if predictor < threshold and label == "YES":
            fn += 1
    # print
    # threshold,tp,tn,fp,fn,float(tp+tn)/total,float(tp)/(tp+fp),float(tp)/(tp+fn)
    acc = float(tp + tn)/total
    pre = float(tp)/(tp + fp + 0.000000009) # Avoid the 0 / 0.
    recall = float(tp)/(tp + fn)
    if (tp + fp) > 0 and (tp + fn) > 0:
    	print ("%.3f\t %d\t %d\t %d\t %d\t %.3f\t %.3f\t %.3f"  %(threshold,tp,tn,fp,fn,acc,pre,recall))
