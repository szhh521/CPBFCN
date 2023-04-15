import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import argparse
import math
import numpy as np

def seq_compar(all_seq,tomtom_seq):
    other_seq = []
    tomtom_filter = []
    tomtom_motif = []
    all_filter = []
    all_motif = []
    for l1 in all_seq:
        tmp1 = l1.split('\t')
        all_filter.append(tmp1[1])
        all_motif.append(tmp1[2])
    for l2 in tomtom_seq:
        tmp2 = l2.split('\t')
        tomtom_filter.append(tmp2[0])
        tomtom_motif.append(tmp2[1])
    for l3 in all_seq:
        tmp3 = l3.split('\t')
        if tmp3[1] in tomtom_filter:
            continue
        else:
            other_seq.append(tmp3[1] + '\t' + tmp3[2])
    return other_seq

def str_compar(str1,str2):
    count = 0
    for l1,l2 in zip(str1,str2):
        if l1 == l2:
            count = count + 1
    return count

def real_find(seq,data):
    count = ''
    for l1 in range(len(data)):
        tmp_result = []
        tmp_pos = []
        for l2 in range(len(data[l1]) - len(seq)):
            if l2 + len(seq) > len(data[l1]):
                break
            else:
                com_result = str_compar(data[l1][l2:l2 + len(seq)], seq)
                if com_result >= int(math.ceil(len(seq)/2)):
                    #tmp_result.append(com_result)
                    #tmp_pos.append(str(l2 + len(seq)/2))
                    count = count + str(l1) + ':' + str(l2 + len(seq)/2) + ';'
    return count

def motif_count_rbp(seq1,pos_path):
    pos_data = []
    for pinfo in open(pos_path):
        if '>' in pinfo:
            continue
        else:
            pos_data.append(pinfo[:-1].replace('T','U'))
    count1 = real_find(seq1,pos_data)
    return count1

def dist_data_process(locdata1):
    loc1 = []
    for l1 in locdata1:
        tmp1 = l1.split(';')
        for l3 in tmp1[:-2]:
            t1 = l3.split(':')
            loc1.append(float(t1[1]))
    return loc1

def create_fig(rbp_path,query_seq,filter,fig_path):
    meme_motif_seq = query_seq
    count1 = motif_count_rbp(meme_motif_seq, rbp_path)
    loc1 = dist_data_process([count1])
    df1 = gaussian_kde(loc1)
    df1.covariance_factor = lambda: .25
    df1._compute_covariance()
    xs1 = np.linspace(0, 201)
    test = df1(xs1)
    plt.plot(xs1, df1(xs1))
    plt.plot([0, 0], [0, np.max(test)], color='green', linestyle='--')
    plt.plot([201, 201], [0, np.max(test)], color='green', linestyle='--')
    plt.plot([100, 100], [0, np.max(test)], color='red', linestyle='--')
    plt.plot([150, 150], [0, np.max(test)], color='blue', linestyle='--')
    plt.plot([50, 50], [0, np.max(test)], color='blue', linestyle='--')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Position', fontsize=18)
    plt.ylabel('Density', fontsize=18)
    plt.title(filter,fontsize=18)
    plt.savefig(fig_path + '\\' + filter + '.png',dpi=300,bbox_inches ="tight",transparent = True)
    plt.cla()

fig_path = 'G:\CPBPFCN-Result\\201_fig'
motif_seq_EIF4A3 = []
motif_seq_FOX2 = []
motif_seq_IGF2BP1 = []
motif_seq_IGF2BP2 = []
motif_seq_IGF2BP3 = []
motif_seq_ZC3H7B = []
for con in open('201_sixdatasets_all_motif_seq'):
    if 'EIF4A3' in con:
        motif_seq_EIF4A3.append(con[:-1])
    elif 'FOX2' in con:
        motif_seq_FOX2.append(con[:-1])
    elif 'IGF2BP1' in con:
        motif_seq_IGF2BP1.append(con[:-1])
    elif 'IGF2BP2' in con:
        motif_seq_IGF2BP2.append(con[:-1])
    elif 'IGF2BP3' in con:
        motif_seq_IGF2BP3.append(con[:-1])
    elif 'ZC3H7B' in con:
        motif_seq_ZC3H7B.append(con[:-1])

for pname in os.listdir(fig_path):
    print('test')
    pro_fig_path = fig_path + '\\' + pname
    tomtom_file_path = 'G:\CPBPFCN-Result\\201_dist_data' + '\\' + pname + '\\tomtom\\tomtom.txt'
    tomtom_seq = []
    dist_seq = []
    for tom in open(tomtom_file_path):
        if tom[0] == 'Q':
            continue
        else:
            tmp = tom[:-1].split(' ')
            tomtom_seq.append(tmp[0] + '\t' + tmp[7])
    if pname == 'EIF4A3':
        other_seq = seq_compar(motif_seq_EIF4A3, tomtom_seq)
        rbp_path = 'G:\CPBPFCN-Result\data\\rbp_dataset\EIF4A3\\201_new.fa'
        for oseq in other_seq:
            filter = oseq.split('\t')[0]
            query_seq = oseq.split('\t')[1]
            create_fig(rbp_path,query_seq,filter,pro_fig_path)
    elif pname == 'FOX2':
        other_seq = seq_compar(motif_seq_FOX2, tomtom_seq)
        rbp_path = 'G:\CPBPFCN-Result\data\\rbp_dataset\FOX2\\201_new.fa'
        for oseq in other_seq:
            filter = oseq.split('\t')[0]
            query_seq = oseq.split('\t')[1]
            create_fig(rbp_path, query_seq, filter, pro_fig_path)
    elif pname == 'IGF2BP1':
        other_seq = seq_compar(motif_seq_IGF2BP1, tomtom_seq)
        rbp_path = 'G:\CPBPFCN-Result\data\\rbp_dataset\IGF2BP1\\201_new.fa'
        for oseq in other_seq:
            filter = oseq.split('\t')[0]
            query_seq = oseq.split('\t')[1]
            create_fig(rbp_path, query_seq, filter, pro_fig_path)
    elif pname == 'IGF2BP2':
        other_seq = seq_compar(motif_seq_IGF2BP2, tomtom_seq)
        rbp_path = 'G:\CPBPFCN-Result\data\\rbp_dataset\IGF2BP2\\201_new.fa'
        for oseq in other_seq:
            filter = oseq.split('\t')[0]
            query_seq = oseq.split('\t')[1]
            create_fig(rbp_path, query_seq, filter, pro_fig_path)
    elif pname == 'IGF2BP3' in con:
        other_seq = seq_compar(motif_seq_IGF2BP3, tomtom_seq)
        rbp_path = 'G:\CPBPFCN-Result\data\\rbp_dataset\IGF2BP3\\201_new.fa'
        for oseq in other_seq:
            filter = oseq.split('\t')[0]
            query_seq = oseq.split('\t')[1]
            create_fig(rbp_path, query_seq, filter, pro_fig_path)
    elif pname == 'ZC3H7B':
        other_seq = seq_compar(motif_seq_ZC3H7B, tomtom_seq)
        rbp_path = 'G:\CPBPFCN-Result\data\\rbp_dataset\ZC3H7B\\201_new.fa'
        for oseq in other_seq:
            filter = oseq.split('\t')[0]
            query_seq = oseq.split('\t')[1]
            create_fig(rbp_path, query_seq, filter, pro_fig_path)
