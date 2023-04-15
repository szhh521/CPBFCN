import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import argparse
import math
import numpy as np

def data_process(init_data):
    str1 = init_data.split('#')
    start = str1[0].split(',')
    end = str1[-1].split(',')
    ints = []
    inte = []
    for inds, inde in zip(start[1:],end[1:]):
        ints.append(int(inds))
        inte.append(int(inde))
    sort_data = sorted(enumerate(ints),key=lambda x:x[1])
    sort_start = [ints[1] for ints in sort_data]
    sort_ind = [ints[0] for ints in sort_data]
    sort_end = [inte[sidx] for sidx in sort_ind]
    start_final = []
    end_final = []
    tmp_start = sort_start[0]
    tmp_end = sort_end[0]
    for idx in range(1,len(sort_start)):
        if idx < len(sort_start) - 1:
            if sort_start[idx] > tmp_end:
                start_final.append(tmp_start)
                end_final.append(tmp_end)
                tmp_start = sort_start[idx]
                tmp_end = sort_end[idx]
            elif sort_start[idx] <= tmp_end:
                if sort_start[idx] >= tmp_start and sort_end[idx] <= tmp_end:
                    tmp_start = sort_start[idx - 1]
                    tmp_end = sort_end[idx - 1]
                elif sort_start[idx] >= tmp_start and sort_end[idx] > tmp_end:
                    tmp_start = sort_start[idx - 1]
                    tmp_end = sort_end[idx]
        elif idx == len(sort_start) - 1:
            if sort_start[idx] > tmp_end:
                start_final.append(tmp_start)
                end_final.append(tmp_end)
                start_final.append(sort_start[idx])
                end_final.append(sort_end[idx])
            elif sort_start[idx] <= tmp_end:
                if sort_start[idx] >= tmp_start and sort_end[idx] <= tmp_end:
                    start_final.append(sort_start[idx - 1])
                    end_final.append(sort_end[idx - 1])
                elif sort_start[idx] >= tmp_start and sort_end[idx] > tmp_end:
                    start_final.append(sort_start[idx - 1])
                    end_final.append(sort_end[idx])
    return start_final,end_final

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
    parser.add_argument('-r', dest='rbp_path', type=str, default='G:\CPBPFCN-Result\\all_data\\101\TIA1\positive')
    parser.add_argument('-c', dest='circ_path', type=str, default='G:\CPBPFCN-Result\data\human_hg19_circRNAs_putative_spliced_sequence.fa')
    parser.add_argument('-n', dest='circ_name', type=str, default='hsa_circ_0000002')
    parser.add_argument('-cm', dest='cfcn_motif', type=str, default='GAAGGCGCUA')
    parser.add_argument('-mm', dest='meme_motif', type=str, default='GAAGGAG')
    parser.add_argument('-s', dest='sponge_path', type=str, default='G:\circRNA_related_data\\circRNA_rbp_sponge_loc')
    return parser.parse_args()

def main():
    params = get_parm()
    meme_motif_seq = params.meme_motif
    model_motif_seq = params.cfcn_motif
    circ_name = params.circ_name
    circ_path = params.circ_path
    sponge_path = params.sponge_path
    circ_hg19 = []
    name = []
    seqdata = []
    sponge_data = []
    for data in open(circ_path):
        if data[0] == '>':
            name.append(data[1:17])
        else:
            seqdata.append(data[:-1].replace('T', 'U'))
    for na, seq in zip(name, seqdata):
        circ_hg19.append((na, seq))
    circ_hg19 = dict(circ_hg19)
    ###RBP sponge data
    for spond in open(sponge_path):
        tmp_sponge = spond[:-1].split('\t')
        sponge_data.append((tmp_sponge[0],tmp_sponge[1]))
    ###
    sponge_data = dict(sponge_data)
    if circ_name not in sponge_data:
        print('circ not exist', circ_name)
    else:
        rbps_loc_start, rbps_loc_end = data_process(sponge_data.get(circ_name))
        count1, count2 = motif_count_circrna(meme_motif_seq, model_motif_seq, circ_hg19.get(circ_name))
        loc1, loc2 = dist_data_process([count1], [count2])
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
        #plt.plot([math.ceil(seq_len / 2), math.ceil(seq_len / 2)], [0, np.max(test)],color='red', linestyle='--')
        for inds, inde in zip(rbps_loc_start, rbps_loc_end):
            plt.stackplot([inds, inde], [np.max(test), np.max(test)], color='r',alpha=0.3)
        plt.legend(loc='upper center',ncol=1,fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel('Position', fontsize=18)
        plt.ylabel('Density', fontsize=18)
        plt.show()
if __name__ == "__main__":
    main()