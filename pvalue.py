import os

import scipy,json
import numpy as np
from tqdm import tqdm
from scipy.stats import ttest_ind,levene

def calculate_fmax(preds, labels):
    preds = np.round(preds, 2)
    labels = labels.astype(np.int32)
    f_max = 0
    p_max = 0
    r_max = 0
    sp_max = 0
    t_max = 0
    for t in range(1, 100):
        threshold = t / 100.0
        predictions = (preds > threshold).astype(np.int32)
        tp = np.sum(predictions * labels)
        fp = np.sum(predictions) - tp
        fn = np.sum(labels) - tp
        sn = tp / (1.0 * np.sum(labels))
        sp = np.sum((predictions ^ 1) * (labels ^ 1))
        sp /= 1.0 * np.sum(labels ^ 1)
        fpr = 1 - sp
        precision = tp / (1.0 * (tp + fp))
        recall = tp / (1.0 * (tp + fn))
        f = 2 * precision * recall / (precision + recall)
        if f_max < f:
            f_max = f
            p_max = precision
            r_max = recall
            sp_max = sp
            t_max = threshold
    return f_max


CFAGO_pred_MF = np.load('./data/Human/p-value/CFAGO-MF_pred.npy')#>0.5)+0.0
CFAGO_label_MF = np.load('./data/Human/p-value/CFAGO-MF_label.npy')
CFAGO_error_MF = np.sum((CFAGO_pred_MF - CFAGO_label_MF)**2 , axis=1)
CFAGO_Fmax_MF = np.array([calculate_fmax(CFAGO_pred_MF[i,:],CFAGO_label_MF[i,:]) for i in range(CFAGO_pred_MF.shape[0])])

CFAGO_pred_BP = np.load('./data/Human/p-value/CFAGO-BP_pred.npy')#>0.5)+0.0
CFAGO_label_BP = np.load('./data/Human/p-value/CFAGO-BP_label.npy')
CFAGO_error_BP = np.sum((CFAGO_pred_BP - CFAGO_label_BP)**2 , axis=1)
CFAGO_Fmax_BP = np.array([calculate_fmax(CFAGO_pred_BP[i,:],CFAGO_label_BP[i,:]) for i in range(CFAGO_pred_BP.shape[0])])

CFAGO_pred_CC = np.load('./data/Human/p-value/CFAGO-CC_pred.npy')#>0.5)+0.0
CFAGO_label_CC = np.load('./data/Human/p-value/CFAGO-CC_label.npy')
CFAGO_error_CC = np.sum((CFAGO_pred_CC - CFAGO_label_CC)**2 , axis=1)
CFAGO_Fmax_CC = np.array([calculate_fmax(CFAGO_pred_CC[i,:],CFAGO_label_CC[i,:]) for i in range(CFAGO_pred_CC.shape[0])])

MultiPredGO_pred_MF = np.load('./data/Human/p-value/MultiPredGO-MF_pred.npy')#>0.5)+0.0
MultiPredGO_label_MF = np.load('./data/Human/p-value/MultiPredGO-MF_label.npy')
MultiPredGO_error_MF = np.sum((MultiPredGO_pred_MF - MultiPredGO_label_MF)**2 , axis=1)
MultiPredGO_Fmax_MF = np.array([calculate_fmax(MultiPredGO_pred_MF[i,:],MultiPredGO_label_MF[i,:]) for i in range(MultiPredGO_pred_MF.shape[0])])

MultiPredGO_pred_BP = np.load('./data/Human/p-value/MultiPredGO-BP_pred.npy')#>0.5)+0.0
MultiPredGO_label_BP = np.load('./data/Human/p-value/MultiPredGO-BP_label.npy')
MultiPredGO_error_BP = np.sum((MultiPredGO_pred_BP - MultiPredGO_label_BP)**2 , axis=1)
MultiPredGO_Fmax_BP = np.array([calculate_fmax(MultiPredGO_pred_BP[i,:],MultiPredGO_label_BP[i,:]) for i in range(MultiPredGO_pred_BP.shape[0])])

MultiPredGO_pred_CC = np.load('./data/Human/p-value/MultiPredGO-CC_pred.npy')#>0.5)+0.0
MultiPredGO_label_CC = np.load('./data/Human/p-value/MultiPredGO-CC_label.npy')
MultiPredGO_error_CC = np.sum((MultiPredGO_pred_CC - MultiPredGO_label_CC)**2 , axis=1)
MultiPredGO_Fmax_CC = np.array([calculate_fmax(MultiPredGO_pred_CC[i,:],MultiPredGO_label_CC[i,:]) for i in range(MultiPredGO_pred_CC.shape[0])])


DeepFusionGO_pred_MF = np.load('./data/Human/p-value/DeepFusionGO-MF_pred.npy')#>0.5)+0.0
DeepFusionGO_label_MF = np.load('./data/Human/p-value/DeepFusionGO-MF_label.npy')
DeepFusionGO_error_MF = np.sum((DeepFusionGO_pred_MF - DeepFusionGO_label_MF)**2 , axis=1)
DeepFusionGO_Fmax_MF = np.array([calculate_fmax(DeepFusionGO_pred_MF[i,:],DeepFusionGO_label_MF[i,:]) for i in range(DeepFusionGO_pred_MF.shape[0])])

DeepFusionGO_pred_BP = np.load('./data/Human/p-value/DeepFusionGO-BP_pred.npy')#>0.5)+0.0
DeepFusionGO_label_BP = np.load('./data/Human/p-value/DeepFusionGO-BP_label.npy')
DeepFusionGO_error_BP = np.sum((DeepFusionGO_pred_BP - DeepFusionGO_label_BP)**2 , axis=1)
DeepFusionGO_Fmax_BP = np.array([calculate_fmax(DeepFusionGO_pred_BP[i,:],DeepFusionGO_label_BP[i,:]) for i in range(DeepFusionGO_pred_BP.shape[0])])

DeepFusionGO_pred_CC = np.load('./data/Human/p-value/DeepFusionGO-CC_pred.npy')#>0.5)+0.0
DeepFusionGO_label_CC = np.load('./data/Human/p-value/DeepFusionGO-CC_label.npy')
DeepFusionGO_error_CC = np.sum((DeepFusionGO_pred_CC - DeepFusionGO_label_CC)**2 , axis=1)
DeepFusionGO_Fmax_CC = np.array([calculate_fmax(DeepFusionGO_pred_CC[i,:],DeepFusionGO_label_CC[i,:]) for i in range(DeepFusionGO_pred_CC.shape[0])])


DeepGraphGO_pred_MF = np.load('./data/Human/p-value/DeepGraphGO-MF_pred.npy')#>0.5)+0.0
DeepGraphGO_label_MF = np.load('./data/Human/p-value/DeepGraphGO-MF_label.npy')
DeepGraphGO_error_MF = np.sum((DeepGraphGO_pred_MF - DeepGraphGO_label_MF)**2 , axis=1)
DeepGraphGO_Fmax_MF = np.array([calculate_fmax(DeepGraphGO_pred_MF[i,:],DeepGraphGO_label_MF[i,:]) for i in range(DeepGraphGO_pred_MF.shape[0])])

DeepGraphGO_pred_BP = np.load('./data/Human/p-value/DeepGraphGO-BP_pred.npy')#>0.5)+0.0
DeepGraphGO_label_BP = np.load('./data/Human/p-value/DeepGraphGO-BP_label.npy')
DeepGraphGO_error_BP = np.sum((DeepGraphGO_pred_BP - DeepGraphGO_label_BP)**2 , axis=1)
DeepGraphGO_Fmax_BP = np.array([calculate_fmax(DeepGraphGO_pred_BP[i,:],DeepGraphGO_label_BP[i,:]) for i in range(DeepGraphGO_pred_BP.shape[0])])

DeepGraphGO_pred_CC = np.load('./data/Human/p-value/DeepGraphGO-CC_pred.npy')#>0.5)+0.0
DeepGraphGO_label_CC = np.load('./data/Human/p-value/DeepGraphGO-CC_label.npy')
DeepGraphGO_error_CC = np.sum((DeepGraphGO_pred_CC - DeepGraphGO_label_CC)**2 , axis=1)
DeepGraphGO_Fmax_CC = np.array([calculate_fmax(DeepGraphGO_pred_CC[i,:],DeepGraphGO_label_CC[i,:]) for i in range(DeepGraphGO_pred_CC.shape[0])])


HEAL_pred_MF = np.load('./data/Human/p-value/HEAL-MF_pred.npy')#>0.5)+0.0
HEAL_label_MF = np.load('./data/Human/p-value/HEAL-MF_label.npy')
HEAL_error_MF = np.sum((HEAL_pred_MF - HEAL_label_MF)**2 , axis=1)
HEAL_Fmax_MF = np.array([calculate_fmax(HEAL_pred_MF[i,:],HEAL_label_MF[i,:]) for i in range(HEAL_pred_MF.shape[0])])

HEAL_pred_BP = np.load('./data/Human/p-value/HEAL-BP_pred.npy')#>0.5)+0.0
HEAL_label_BP = np.load('./data/Human/p-value/HEAL-BP_label.npy')
HEAL_error_BP = np.sum((HEAL_pred_BP - HEAL_label_BP)**2 , axis=1)
HEAL_Fmax_BP = np.array([calculate_fmax(HEAL_pred_BP[i,:],HEAL_label_BP[i,:]) for i in range(HEAL_pred_BP.shape[0])])

HEAL_pred_CC = np.load('./data/Human/p-value/HEAL-CC_pred.npy')#>0.5)+0.0
HEAL_label_CC = np.load('./data/Human/p-value/HEAL-CC_label.npy')
HEAL_error_CC = np.sum((HEAL_pred_CC - HEAL_label_CC)**2 , axis=1)
HEAL_Fmax_CC = np.array([calculate_fmax(HEAL_pred_CC[i,:],HEAL_label_CC[i,:]) for i in range(HEAL_pred_CC.shape[0])])


GATGO_pred_MF = np.load('./data/Human/p-value/GATGO-MF_pred.npy')#>0.5)+0.0
GATGO_label_MF = np.load('./data/Human/p-value/GATGO-MF_label.npy')
GATGO_error_MF = np.sum((GATGO_pred_MF - GATGO_label_MF)**2 , axis=1)
GATGO_Fmax_MF = np.array([calculate_fmax(GATGO_pred_MF[i,:],GATGO_label_MF[i,:]) for i in range(GATGO_pred_MF.shape[0])])

GATGO_pred_BP = np.load('./data/Human/p-value/GATGO-BP_pred.npy')#>0.5)+0.0
GATGO_label_BP = np.load('./data/Human/p-value/GATGO-BP_label.npy')
GATGO_error_BP = np.sum((GATGO_pred_BP - GATGO_label_BP)**2 , axis=1)
GATGO_Fmax_BP = np.array([calculate_fmax(GATGO_pred_BP[i,:],GATGO_label_BP[i,:]) for i in range(GATGO_pred_BP.shape[0])])

GATGO_pred_CC = np.load('./data/Human/p-value/GATGO-CC_pred.npy')#>0.5)+0.0
GATGO_label_CC = np.load('./data/Human/p-value/GATGO-CC_label.npy')
GATGO_error_CC = np.sum((GATGO_pred_CC - GATGO_label_CC)**2 , axis=1)
GATGO_Fmax_CC = np.array([calculate_fmax(GATGO_pred_CC[i,:],GATGO_label_CC[i,:]) for i in range(GATGO_pred_CC.shape[0])])

DeepFRI_pred_MF = np.load('./data/Human/p-value/DeepFRI-MF_pred.npy')#>0.5)+0.0
DeepFRI_label_MF = np.load('./data/Human/p-value/DeepFRI-MF_label.npy')
DeepFRI_error_MF = np.sum((DeepFRI_pred_MF - DeepFRI_label_MF)**2 , axis=1)
DeepFRI_Fmax_MF = np.array([calculate_fmax(DeepFRI_pred_MF[i,:],DeepFRI_label_MF[i,:]) for i in range(DeepFRI_pred_MF.shape[0])])

DeepFRI_pred_BP = np.load('./data/Human/p-value/DeepFRI-BP_pred.npy')#>0.5)+0.0
DeepFRI_label_BP = np.load('./data/Human/p-value/DeepFRI-BP_label.npy')
DeepFRI_error_BP = np.sum((DeepFRI_pred_BP - DeepFRI_label_BP)**2 , axis=1)
DeepFRI_Fmax_BP = np.array([calculate_fmax(DeepFRI_pred_BP[i,:],DeepFRI_label_BP[i,:]) for i in range(DeepFRI_pred_BP.shape[0])])

DeepFRI_pred_CC = np.load('./data/Human/p-value/DeepFRI-CC_pred.npy')#>0.5)+0.0
DeepFRI_label_CC = np.load('./data/Human/p-value/DeepFRI-CC_label.npy')
DeepFRI_error_CC = np.sum((DeepFRI_pred_CC - DeepFRI_label_CC)**2 , axis=1)
DeepFRI_Fmax_CC = np.array([calculate_fmax(DeepFRI_pred_CC[i,:],DeepFRI_label_CC[i,:]) for i in range(DeepFRI_pred_CC.shape[0])])

MIF2GO_pred_MF = np.load('./data/Human/p-value/MIF2GO-MF_pred.npy')#>0.5)+0.0
MIF2GO_label_MF = np.load('./data/Human/p-value/MIF2GO-MF_label.npy')
MIF2GO_error_MF = np.sum((MIF2GO_pred_MF - MIF2GO_label_MF)**2 , axis=1)
MIF2GO_Fmax_MF = np.array([calculate_fmax(MIF2GO_pred_MF[i,:],MIF2GO_label_MF[i,:]) for i in range(MIF2GO_pred_MF.shape[0])])

MIF2GO_pred_BP = np.load('./data/Human/p-value/MIF2GO-BP_pred.npy')#>0.5)+0.0
MIF2GO_label_BP = np.load('./data/Human/p-value/MIF2GO-BP_label.npy')
MIF2GO_error_BP = np.sum((MIF2GO_pred_BP - MIF2GO_label_BP)**2 , axis=1)
MIF2GO_Fmax_BP = np.array([calculate_fmax(MIF2GO_pred_BP[i,:],MIF2GO_label_BP[i,:]) for i in range(MIF2GO_pred_BP.shape[0])])

MIF2GO_pred_CC = np.load('./data/Human/p-value/MIF2GO-CC_pred.npy')#>0.5)+0.0
MIF2GO_label_CC = np.load('./data/Human/p-value/MIF2GO-CC_label.npy')
MIF2GO_error_CC = np.sum((MIF2GO_pred_CC - MIF2GO_label_CC)**2 , axis=1)
MIF2GO_Fmax_CC = np.array([calculate_fmax(MIF2GO_pred_CC[i,:],MIF2GO_label_CC[i,:]) for i in range(MIF2GO_pred_CC.shape[0])])






DeepFRI_p_value_MF = ttest_ind(MIF2GO_Fmax_MF, DeepFRI_Fmax_MF,equal_var=levene(MIF2GO_Fmax_MF, DeepFRI_Fmax_MF).pvalue>0.05).pvalue
DeepFRI_p_value_BP = ttest_ind(MIF2GO_Fmax_BP, DeepFRI_Fmax_BP,equal_var=levene(MIF2GO_Fmax_BP, DeepFRI_Fmax_BP).pvalue>0.05).pvalue
DeepFRI_p_value_CC = ttest_ind(MIF2GO_Fmax_CC, DeepFRI_Fmax_CC,equal_var=levene(MIF2GO_Fmax_CC, DeepFRI_Fmax_CC).pvalue>0.05).pvalue


GATGO_p_value_MF = ttest_ind(MIF2GO_Fmax_MF,GATGO_Fmax_MF,equal_var=levene(MIF2GO_Fmax_MF,GATGO_Fmax_MF).pvalue>0.05).pvalue
GATGO_p_value_BP = ttest_ind(MIF2GO_Fmax_BP,GATGO_Fmax_BP,equal_var=levene(MIF2GO_Fmax_BP,GATGO_Fmax_BP).pvalue>0.05).pvalue
GATGO_p_value_CC = ttest_ind(MIF2GO_Fmax_CC,GATGO_Fmax_CC,equal_var=levene(MIF2GO_Fmax_CC,GATGO_Fmax_CC).pvalue>0.05).pvalue



HEAL_p_value_MF = ttest_ind(MIF2GO_Fmax_MF, HEAL_Fmax_MF,equal_var=levene(MIF2GO_Fmax_MF, HEAL_Fmax_MF).pvalue>0.05).pvalue
HEAL_p_value_BP = ttest_ind(MIF2GO_Fmax_BP, HEAL_Fmax_BP,equal_var=levene(MIF2GO_Fmax_BP, HEAL_Fmax_BP).pvalue>0.05).pvalue
HEAL_p_value_CC = ttest_ind(MIF2GO_Fmax_CC, HEAL_Fmax_CC,equal_var=levene(MIF2GO_Fmax_CC, HEAL_Fmax_CC).pvalue>0.05).pvalue


DeepGraphGO_p_value_MF = ttest_ind(MIF2GO_Fmax_MF, DeepGraphGO_Fmax_MF,equal_var=levene(MIF2GO_Fmax_MF, DeepGraphGO_Fmax_MF).pvalue>0.05).pvalue
DeepGraphGO_p_value_BP = ttest_ind(MIF2GO_Fmax_BP, DeepGraphGO_Fmax_BP,equal_var=levene(MIF2GO_Fmax_BP, DeepGraphGO_Fmax_BP).pvalue>0.05).pvalue
DeepGraphGO_p_value_CC = ttest_ind(MIF2GO_Fmax_CC, DeepGraphGO_Fmax_CC,equal_var=levene(MIF2GO_Fmax_CC, DeepGraphGO_Fmax_CC).pvalue>0.05).pvalue


DeepFusionGO_p_value_MF = ttest_ind(MIF2GO_Fmax_MF, DeepFusionGO_Fmax_MF,equal_var=levene(MIF2GO_Fmax_MF, DeepFusionGO_Fmax_MF).pvalue>0.05).pvalue
DeepFusionGO_p_value_BP = ttest_ind(MIF2GO_Fmax_BP, DeepFusionGO_Fmax_BP,equal_var=levene(MIF2GO_Fmax_BP, DeepFusionGO_Fmax_BP).pvalue>0.05).pvalue
DeepFusionGO_p_value_CC = ttest_ind(MIF2GO_Fmax_CC, DeepFusionGO_Fmax_CC,equal_var=levene(MIF2GO_Fmax_CC, DeepFusionGO_Fmax_CC).pvalue>0.05).pvalue



MultiPredGO_p_value_MF = ttest_ind(MIF2GO_Fmax_MF, MultiPredGO_Fmax_MF,equal_var=levene(MIF2GO_Fmax_MF, MultiPredGO_Fmax_MF).pvalue>0.05).pvalue
MultiPredGO_p_value_BP = ttest_ind(MIF2GO_Fmax_BP, MultiPredGO_Fmax_BP,equal_var=levene(MIF2GO_Fmax_BP, MultiPredGO_Fmax_BP).pvalue>0.05).pvalue
MultiPredGO_p_value_CC = ttest_ind(MIF2GO_Fmax_CC, MultiPredGO_Fmax_CC,equal_var=levene(MIF2GO_Fmax_CC, MultiPredGO_Fmax_CC).pvalue>0.05).pvalue


CFAGO_p_value_MF = ttest_ind(MIF2GO_Fmax_MF, CFAGO_Fmax_MF,equal_var=levene(MIF2GO_Fmax_MF, CFAGO_Fmax_MF).pvalue>0.05).pvalue
CFAGO_p_value_BP = ttest_ind(MIF2GO_Fmax_BP, CFAGO_Fmax_BP,equal_var=levene(MIF2GO_Fmax_BP, CFAGO_Fmax_BP).pvalue>0.05).pvalue
CFAGO_p_value_CC = ttest_ind(MIF2GO_Fmax_CC, CFAGO_Fmax_CC,equal_var=levene(MIF2GO_Fmax_CC, CFAGO_Fmax_CC).pvalue>0.05).pvalue


print()
