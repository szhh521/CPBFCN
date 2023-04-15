import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import argparse
import math
import numpy as np

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

def motif_count_rbp(seq1,seq2,pos_path):
    pos_data = []
    for pinfo in open(pos_path):
        if '>' in pinfo:
            continue
        else:
            pos_data.append(pinfo[:-1].replace('T','U'))
    count1 = real_find(seq1,pos_data)
    count2 = real_find(seq2,pos_data)
    return count1,count2

def motif_count_circrna(seq1,seq2,pos_seq):
    count1 = real_find(seq1,[pos_seq])
    count2 = real_find(seq2,[pos_seq])
    return count1,count2

def dist_data_process(locdata1,locdata2):
    loc1 = []
    for l1 in locdata1:
        tmp1 = l1.split(';')
        for l3 in tmp1[:-2]:
            t1 = l3.split(':')
            loc1.append(float(t1[1]))
    loc2 = []
    for l2 in locdata2:
        tmp2 = l2.split(';')
        for l3 in tmp2[:-2]:
            t1 = l3.split(':')
            loc2.append(float(t1[1]))
    return loc1,loc2

def get_parm():
    parser = argparse.ArgumentParser(description='motif dist')
    parser.add_argument('-t',dest='data_type',type=str,default='RBP')
    parser.add_argument('-r', dest='rbp_path', type=str, default='G:\CPBPFCN-Result\\all_data\\101\TIA1\positive')
    parser.add_argument('-c', dest='circ_path', type=str, default='G:\CPBPFCN-Result\data\human_hg19_circRNAs_putative_spliced_sequence.fa')
    parser.add_argument('-n', dest='circ_name', type=str, default='hsa_circ_0000001')
    parser.add_argument('-cm', dest='cfcn_motif', type=str, default='UUUUUCUUUU')
    parser.add_argument('-mm', dest='meme_motif', type=str, default='UUUUUUC')
    return parser.parse_args()

def main():
    params = get_parm()
    data_type = params.data_type
    if data_type == 'RBP':
        meme_motif_seq = params.meme_motif
        model_motif_seq = params.cfcn_motif
        count1, count2 = motif_count_rbp(meme_motif_seq, model_motif_seq, params.rbp_path)
        loc1,loc2 = dist_data_process([count1],[count2])
        df1 = gaussian_kde(loc1)
        df1.covariance_factor = lambda: .25
        df1._compute_covariance()
        xs1 = np.linspace(0, 101)
        df2 = gaussian_kde(loc2)
        df2.covariance_factor = lambda: .25
        df2._compute_covariance()
        xs2 = np.linspace(0, 101)
        test = df1(xs1)
        plt.plot(xs1, df1(xs1), label='Meme motif')
        plt.plot(xs2, df2(xs2), label='Model motif')
        plt.plot([0, 0], [0, np.max(test)], color='green', linestyle='--')
        plt.plot([101, 101], [0, np.max(test)], color='green', linestyle='--')
        plt.plot([50, 50], [0, np.max(test)], color='red', linestyle='--')
        plt.legend(loc='lower center',fontsize=16,ncol=2)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel('Position', fontsize=18)
        plt.ylabel('Density', fontsize=18)
        plt.show()
    elif data_type == 'CRNA':
        meme_motif_seq = params.meme_motif
        model_motif_seq = params.cfcn_motif
        circ_name = params.circ_name
        circ_path = params.circ_path
        circ_hg19 = []
        name = []
        seqdata = []
        for data in open(circ_path):
            if data[0] == '>':
                name.append(data[1:17])
            else:
                seqdata.append(data[:-1].replace('T','U'))
        for na,seq in zip(name,seqdata):
            circ_hg19.append((na,seq))
        circ_hg19 = dict(circ_hg19)
        count1, count2 = motif_count_circrna(meme_motif_seq, model_motif_seq,circ_hg19.get(circ_name))
        loc1,loc2 = dist_data_process([count1],[count2])
        seq_len = len(circ_hg19.get(circ_name))
        df1 = gaussian_kde(loc1)
        df1.covariance_factor = lambda: .25
        df1._compute_covariance()
        xs1 = np.linspace(0, seq_len)
        df2 = gaussian_kde(loc2)
        df2.covariance_factor = lambda: .25
        df2._compute_covariance()
        xs2 = np.linspace(0, seq_len)
        test = df1(xs1)
        plt.plot(xs1, df1(xs1), label='Meme motif')
        plt.plot(xs2, df2(xs2), label='Model motif')
        plt.plot([0, 0], [0, np.max(test)], color='green', linestyle='--')
        plt.plot([seq_len, seq_len], [0, np.max(test)], color='green', linestyle='--')
        plt.plot([math.ceil(seq_len / 2), math.ceil(seq_len / 2)], [0, np.max(test)],
                                        color='red', linestyle='--')
        plt.legend(loc='upper right')
        plt.xlabel('Position', fontsize=15)
        plt.ylabel('Density', fontsize=15)
        plt.show()
if __name__ == "__main__":
    main()
