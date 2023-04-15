import numpy as np
import random
import os
import sys
import argparse
import pandas as pd
from bs4 import BeautifulSoup
from Bio import SeqIO

key_seq = {"A":[1,0,0,0],"C":[0,1,0,0],"G":[0,0,1,0],"T":[0,0,0,1],"N":[0,0,0,0]}

def create_circ_hg19_dict(path):
    circ_hg19_name = []
    circ_hg19_data = []
    for hgid in SeqIO.parse(path + '/human_hg19_circRNAs_putative_spliced_sequence.fa','fasta'):
        temp_id = hgid.description
        circ_hg19_name.append(temp_id[:16])
        circ_hg19_data.append(str(hgid.seq))
    circ_hg19_dict = []
    for ckey, cvalue in zip(circ_hg19_name,circ_hg19_data):
        circ_hg19_dict.append((ckey,cvalue))
    circ_hg19_dict = dict(circ_hg19_dict)
    return circ_hg19_dict

def create_all_circ_dict(path):
    #circ_name = []
    all_circ_dict = []
    circ_info = pd.read_excel(path)
    circ_data = circ_info.values.tolist()
    for circ_id in circ_data:
        all_circ_dict.append((circ_id[0],circ_id[2] + '*' + str(circ_id[3]) + '*' + str(circ_id[4])))
    all_circ_dict = dict(all_circ_dict)
    return all_circ_dict

def get_new_loc(posinfo,alldict):
    circ_name = posinfo[0]
    old_start = posinfo[1]
    old_end = posinfo[2]
    infostr = alldict.get(circ_name)
    print(circ_name)
    print(infostr)
    strand = infostr.split('*')[0]
    origin_length = infostr.split('*')[1]
    spliced_length = infostr.split('*')[2]
    if strand == '+':
        return int(old_start),int(old_end),strand
    elif strand == '-':
        return int(old_start)-(int(origin_length) - int(spliced_length)),int(old_end)-(int(origin_length) - int(spliced_length)), strand

def html2csv(inpath):
    data = []
    list_header = []
    soup = BeautifulSoup(open(inpath),'html.parser')
    header = soup.find_all("table")[0].find('tr')
    for items in header:
        try:
            list_header.append(items.get_text())
        except:
            continue
    HTML_data = soup.find_all("table")[0].find_all('tr')[1:]
    for element in HTML_data:
        sub_data = []
        for sub_element in element:
            try:
                sub_data.append(sub_element.get_text())
            except:
                continue
        data.append(sub_data)
    dataFrame = pd.DataFrame(data = data, columns=list_header)
    return dataFrame

def load_data_from_hg19(data_path,dname,seq_length,hg19_path):
    pos_data_old = []
    pos_label = []
    pos_name = []
    #load positive data
    for pdata in SeqIO.parse(data_path + '/positive', 'fasta'):
            tmp_str = pdata.description
            circ_name = tmp_str.split(' ')[0]
            tmp_loc = tmp_str.split(' ')[1]
            tmp_start = tmp_loc.split(',')[0]
            tmp_end = tmp_loc.split(',')[1]
            start = tmp_start.split(':')[1]
            end = tmp_end.split(':')[1]
            pos_name.append([circ_name,start,end])
            pos_data_old.append(pdata.seq)

    circ_hg19_dict = create_circ_hg19_dict(hg19_path)

    #all_circ_path = '/home/szhen/Downloads/circrna_protein/code-data/xlsdata/all-human-circRNAs.xls'
    #all_circ_dict = create_all_circ_dict(all_circ_path)

    pos_data_new = []
    pos_name_new = []
    label_mid = []
    count = 0
    for pos_index in range(len(pos_name)):
        circ_seq_data = circ_hg19_dict.get(pos_name[pos_index][0])
        #print(str(pos_data_old[pos_index]))
        #print(circ_seq_data.index(pos_data_old[pos_index]))
        try:
            curr_start = circ_seq_data.index(str(pos_data_old[pos_index]))
            if seq_length == 201:
                if curr_start - 50 >= 0 and curr_start - 50 + seq_length <= len(circ_seq_data):
                    tmp_seq = circ_seq_data[curr_start - 50: curr_start - 50 + seq_length]
                    if 'N' not in tmp_seq:
                        pos_data_new.append(circ_seq_data[curr_start - 50: curr_start - 50 + seq_length])
                        pos_name_new.append(pos_name[pos_index])
                        label_mid.append(int(seq_length / 2))
                elif curr_start - 50 < 0 and len(circ_seq_data) > seq_length:
                    tmp_seq = circ_seq_data[:seq_length]
                    if 'N' not in tmp_seq:
                        pos_data_new.append(circ_seq_data[:seq_length])
                        pos_name_new.append(pos_name[pos_index])
                        label_mid.append(curr_start + 50)
                elif curr_start + 101 + 50 > len(circ_seq_data) and len(circ_seq_data) > seq_length:
                    tmp_seq = circ_seq_data[len(circ_seq_data) - seq_length:]
                    if 'N' not in tmp_seq:
                        pos_data_new.append(circ_seq_data[len(circ_seq_data) - seq_length:])
                        pos_name_new.append(pos_name[pos_index])
                        label_mid.append(100 + (100 - (len(circ_seq_data) - (curr_start + 101)) - 50))
                elif seq_length > len(circ_seq_data):
                    count = count + 1
            elif seq_length == 501:
                if curr_start - 200 >= 0 and curr_start - 200 + seq_length <= len(circ_seq_data):
                    tmp_seq = circ_seq_data[curr_start - 200: curr_start - 200 + seq_length]
                    if 'N' not in tmp_seq:
                        pos_data_new.append(circ_seq_data[curr_start - 200: curr_start - 200 + seq_length])
                        pos_name_new.append(pos_name[pos_index])
                        label_mid.append(int(seq_length / 2))
                elif curr_start - 200 < 0 and len(circ_seq_data) > seq_length:
                    tmp_seq = circ_seq_data[:seq_length]
                    if 'N' not in tmp_seq:
                        pos_data_new.append(circ_seq_data[:seq_length])
                        pos_name_new.append(pos_name[pos_index])
                        label_mid.append(curr_start + 50)
                elif curr_start + 101 + 200 > len(circ_seq_data) and len(circ_seq_data) > seq_length:
                    tmp_seq = circ_seq_data[len(circ_seq_data) - seq_length:]
                    if 'N' not in tmp_seq:
                        pos_data_new.append(circ_seq_data[len(circ_seq_data) - seq_length:])
                        pos_name_new.append(pos_name[pos_index])
                        label_mid.append(250 + (250 - (len(circ_seq_data) - (curr_start + 101)) - 50))
                elif seq_length > len(circ_seq_data):
                    count = count + 1
        except:
            print('not found', pos_name[pos_index][0])

    #positive: encoding data and create label
    encoding_pos_data = []
    for ep in pos_data_new:
        ted = []
        for i in ep:
            ted.append(key_seq.get(i))
        encoding_pos_data.append(ted)
    encoding_pos_data = np.array(encoding_pos_data)
    print('Data encoding done')
    #load target info
    xlsname = '/home/szhen/Downloads/circrna_protein/code-data/xlsdata/CircInteractome_RBP_' + dname + '.xls'
    csvname = '/home/szhen/Downloads/circrna_protein/code-data/xlsback/CircInteractome_RBP_' + dname + '_' + str(seq_length) + '.csv'
    #p.save_book_as(file_name=xlsname,dest_file_name = xlsname+'x')
    print(csvname)
    dataFrame = html2csv(xlsname)
    dataFrame.to_csv(csvname)
    protein_info = pd.read_csv(csvname)
    #protein_df = protein_info[0]
    protein_data = protein_info.values.tolist()
    protein_dict = []
    for pro_id in protein_data:
        pro_key = pro_id[2]+'_' + str(pro_id[-4]) + '_' + str(pro_id[-2])
        pro_value = pro_id[-6]
        protein_dict.append((pro_key,pro_value))
    protein_dict = dict(protein_dict)
    for pnid in range(len(pos_name_new)):
        row_label = np.zeros(seq_length)
        pro_key = pos_name_new[pnid][0] + '_' + str(int(pos_name_new[pnid][1]) + 50) + '_' + str(int(pos_name_new[pnid][-1]) - 50)
        #print(protein_dict.get(pro_key))
        try:
            if seq_length == 201:
                row_label[label_mid[pnid] - int(protein_dict.get(pro_key) / 2):label_mid[pnid] + int(protein_dict.get(pro_key) / 2)] = 1
                pos_label.append(row_label)
            elif seq_length == 501:
                row_label[label_mid[pnid] - int(protein_dict.get(pro_key) / 2):label_mid[pnid] + int(protein_dict.get(pro_key) / 2)] = 1
                pos_label.append(row_label)
            #print('test')
        except:
            print('cannot create protein label',pos_name_new[pnid])
    pos_label = np.array(pos_label)
    print('create label done')
    return encoding_pos_data,pos_label, str(seq_length), pos_name_new, label_mid , pos_data_new

def load_data_101(data_path,dname,seq_length,hg19_path):
    pos_data = []
    pos_label = []
    pos_name = []
    #load positive data
    for pdata in SeqIO.parse(data_path + '/positive','fasta'):
            tmp_str = pdata.description
            circ_name = tmp_str.split(' ')[0]
            tmp_loc = tmp_str.split(' ')[1]
            tmp_start = tmp_loc.split(',')[0]
            tmp_end = tmp_loc.split(',')[1]
            start = tmp_start.split(':')[1]
            end = tmp_end.split(':')[1]
            pos_name.append([circ_name,start,end])
            pos_data.append(pdata.seq)
    #positive: encoding data and create label
    encoding_pos_data = []
    for ep in pos_data:
        ted = []
        for i in ep:
            ted.append(key_seq.get(i))
        encoding_pos_data.append(ted)
    encoding_pos_data = np.array(encoding_pos_data)
    print('Data encoding done')
    #load target info
    xlsname = '/home/szhen/Downloads/circ-protein/code-data/xlsdata/CircInteractome_RBP_' + dname + '.xls'
    csvname = '/home/szhen/Downloads/circrna_protein/code-data/xlsback/CircInteractome_RBP_' + dname + '_' + str(seq_length) + '.csv'
    #p.save_book_as(file_name=xlsname,dest_file_name = xlsname+'x')
    print(csvname)
    dataFrame = html2csv(xlsname)
    dataFrame.to_csv(csvname)
    protein_info = pd.read_csv(csvname)
    #protein_df = protein_info[0]
    protein_data = protein_info.values.tolist()
    protein_dict = []
    for pro_id in protein_data:
        pro_key = pro_id[2]+'_' + str(pro_id[-4]) + '_' + str(pro_id[-2])
        pro_value = pro_id[-6]
        protein_dict.append((pro_key,pro_value))
    protein_dict = dict(protein_dict)
    for pnid in pos_name:
        row_label = np.zeros(seq_length)
        pro_key = pnid[0] + '_' + str(int(pnid[1]) + 50) + '_' + str(int(pnid[-1]) - 50)
        #print(protein_dict.get(pro_key))
        try:
            row_label[50 - int(protein_dict.get(pro_key) / 2):50 + int(protein_dict.get(pro_key) / 2)] = 1
            pos_label.append(row_label)
            #print('test')
        except:
            print('cannot create protein label')
    pos_label = np.array(pos_label)
    print('create label done')
    return encoding_pos_data,pos_label,str(seq_length)

def get_parm():
    parser = argparse.ArgumentParser(description="data process.")
    parser.add_argument("-d", dest="data_path", type=str, default='/home/szhen/Downloads/circrna_protein/code-data/hyper_parm')
    parser.add_argument("-l", dest="seq_length", type=int, default=201)
    parser.add_argument("-p", dest="hg19_path", type=str, default='/home/szhen/Downloads/circrna_protein')
    return parser.parse_args()

def main_process():
    params = get_parm()
    seq_data_path = params.data_path
    seq_length = params.seq_length
    hg19_path = params.hg19_path
    #data name
    dname = os.listdir(seq_data_path)
    print('Start Data Process')
    for na in dname:
        print('Now working on:',na)
        if seq_length == 101:
            seqs, labels, name = load_data_101(seq_data_path + '/' + na, na,seq_length, hg19_path)
            np.savez(seq_data_path + '/' + na + '/' + '%s_data.npz' % name, data=seqs, denselabel=labels)
        elif seq_length >= 201:
            seqs, labels, name,pos_name_new, label_mid , pos_data_new = load_data_from_hg19(seq_data_path + '/' + na, na,seq_length, hg19_path)
            if os.path.exists(seq_data_path + '/' + na + '/' + str(name) + '_data_all'):
                os.remove(seq_data_path + '/' + na + '/' + str(name) + '_data_all')
            file_out = open(seq_data_path + '/' + na + '/' + str(name) + '_data_all', 'w')
            for pname, label, pdnew in zip(pos_name_new,label_mid,pos_data_new):
                content_str = pname[0] + '_' + str(pname[1]) + '_' + str(pname[2]) + ' ' + str(label) + ' ' + pdnew
                print(content_str,file=file_out)
            file_out.close()
            np.savez(seq_data_path + '/' + na + '/' + '%s_data.npz' % name, data=seqs, denselabel=labels)

if __name__ == '__main__':
    main_process()

