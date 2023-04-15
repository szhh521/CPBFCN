# -*- coding: utf8 -*-
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import numpy as np
import sys
from seq_logo import *
from seq_motifs import *
import shutil
from datasets import CRIDataSetTrain,CRIDataSetTest
from torch.utils.data import DataLoader

index1 = 0

def get_motif_fig_new(filter_weights, filter_outs, out_dir, seqs, sample_i):
    print ('plot motif fig', out_dir)
    if sample_i:
        print ('sampling')
        seqs = []
        for ind, val in enumerate(seqs):
            if ind in sample_i:
                seqs.append(val)
        filter_outs = filter_outs[sample_i]
    num_filters = filter_weights.shape[0]
    filter_size = 10 # filter_weights.shape[2]

    filters_ic = []
    meme_out = meme_intro(out_dir + '/filters_meme.txt',seqs)


    for f in range(num_filters):
        print ('Filter %d' % f)

        # plot filter parameters as a heatmap
        plot_filter_heat(filter_weights[f,:,:], '%s/filter%d_heat.pdf' % (out_dir,f))

        # write possum motif file
        filter_possum(filter_weights[f,:,:], 'filter%d'%f, '%s/filter%d_possum.txt'%(out_dir,f), False)

        # plot weblogo of high scoring outputs
        plot_filter_logo(filter_outs[:,:, f], filter_size, seqs, '%s/filter%d_logo'%(out_dir,f), maxpct_t=0.5)

        # make a PWM for the filter
        filter_pwm, nsites = make_filter_pwm('%s/filter%d_logo.fa'%(out_dir,f))

        if nsites < 10:
            # no information
            filters_ic.append(0)
        else:
            # compute and save information content
            filters_ic.append(info_content(filter_pwm))

            # add to the meme motif file
            meme_add(meme_out, f, filter_pwm, nsites, False)

    meme_out.close()


    #################################################################
    # annotate filters
    #################################################################
    # run tomtom #-evalue 0.01
    subprocess.call('/home/szhen/meme/bin/tomtom -dist pearson -thresh 0.05 -eps -oc %s/tomtom %s/filters_meme.txt %s' % (out_dir, out_dir, 'Ray2013_rbp_Homo_sapiens.meme'), shell=True)

    #tsv to txt
    tomtom_txt = open(out_dir + '/tomtom/tomtom.txt','w')
    for ltom in open(out_dir + '/tomtom/tomtom.tsv'):
        tmptom = ltom[:-1].split('\t')
        if tmptom[0] != '' and tmptom[0][0] != '#':
            print(' '.join(tmptom),file=tomtom_txt)
    tomtom_txt.close()

    # read in annotations
    filter_names = name_filters(num_filters, '%s/tomtom/tomtom.txt'%out_dir, 'Ray2013_rbp_Homo_sapiens.meme')


    #################################################################
    # print a table of information
    #################################################################
    table_out = open('%s/table.txt'%out_dir, 'w')

    # print header for later panda reading
    header_cols = ('', 'consensus', 'annotation', 'ic', 'mean', 'std')
    print('%3s  %19s  %10s  %5s  %6s  %6s' % header_cols,file=table_out)

    for f in range(num_filters):
        # collapse to a consensus motif
        consensus = filter_motif(filter_weights[f,:,:])

        # grab annotation
        annotation = '.'
        name_pieces = filter_names[f].split('_')
        if len(name_pieces) > 1:
            annotation = name_pieces[1]

        # plot density of filter output scores
        fmean, fstd = plot_score_density(np.ravel(filter_outs[:,:, f]), '%s/filter%d_dens.pdf' % (out_dir,f))

        row_cols = (f, consensus, annotation, filters_ic[f], fmean, fstd)
        print('%-3d  %19s  %10s  %5.2f  %6.4f  %6.4f' % row_cols,file=table_out)

    table_out.close()

    if True:
        new_outs = []
        for val in filter_outs:
            new_outs.append(val.T)
        filter_outs = np.array(new_outs)
        print(filter_outs.shape)
        # plot filter-sequence heatmap
        plot_filter_seq_heat(filter_outs, '%s/filter_seqs.pdf'%out_dir)

def upsample(x, out_size):
    return F.interpolate(x, size=out_size, mode='linear', align_corners=False)


def bn_relu_conv(in_, out_, kernel_size=3, stride=1, bias=False):
    padding = kernel_size // 2
    return nn.Sequential(nn.BatchNorm1d(in_),
                         nn.ReLU(inplace=True),
                         nn.Conv1d(in_, out_, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))


class FCNP(nn.Module):
    """FCN for motif mining"""
    def __init__(self, motiflen=13):
        super(FCNP, self).__init__()
        # encode process
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=128, kernel_size=motiflen)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=5)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        # decode process
        self.blend4 = bn_relu_conv(32, 64, kernel_size=3)
        self.blend3 = bn_relu_conv(64, 128, kernel_size=3)
        self.blend2 = bn_relu_conv(128, 4, kernel_size=3)
        self.blend1 = bn_relu_conv(4, 1, kernel_size=3)
        # general functions
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.1)
        self._init_weights()

    def _init_weights(self):
        """Initialize the new built layers"""
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, data):
        """Construct a new computation graph at each froward"""
        b, _, _ = data.size()
        # encode process
        skip1 = data
        out1 = self.conv1(data)
        out1 = self.relu(out1)
        score = out1
        motif_weight = self.conv1.weight
        out1 = self.pool1(out1)
        out1 = self.dropout(out1)
        skip2 = out1
        out1 = self.conv2(out1)
        out1 = self.relu(out1)
        out1 = self.pool2(out1)
        out1 = self.dropout(out1)
        skip3 = out1
        out1 = self.conv3(out1)
        out1 = self.relu(out1)
        out1 = self.pool3(out1)
        out1 = self.dropout(out1)
        skip4 = out1
        out1 = self.conv4(out1)
        out1 = self.relu(out1)
        #out1 = self.pool4(out1)
        out1 = self.dropout(out1)
        skip5 = out1
        # decode process
        up4 = upsample(skip5, skip4.size()[-1])
        up4 = up4 + skip4
        up4 = self.blend4(up4)
        up3 = upsample(up4, skip3.size()[-1])
        up3 = up3 + skip3
        up3 = self.blend3(up3)
        up2 = upsample(up3, skip2.size()[-1])
        up2 = up2 + skip2
        up2 = self.blend2(up2)
        up1 = upsample(up2, skip1.size()[-1])
        up1 = up1 + skip1
        up1 = self.blend1(up1)
        out_dense = self.sigmoid(up1)
        out_dense = out_dense.view(b, -1)

        return out_dense, score, motif_weight

def main(data_dir,path_npz, name, gpu, motifLen,store_path):
    """Create the model and start the training."""
    if torch.cuda.is_available():
        if len(gpu.split(',')) == 1:
            device = torch.device("cuda:" + gpu)
        else:
            device = torch.device("cuda:" + gpu.split(',')[0])
    else:
        device = torch.device("cpu")

    record_path = dpath + '/' + pname + '/' + str(seq_len) + '_HNML_0.7_FCN_record.txt'
    record = []
    for l2 in open(record_path):
        record.append(l2[:-1].split('\t'))
    all_iou = []
    for l3 in range(len(record)):
        if l3 == 0:
            continue
        elif l3 == len(record) - 1:
            break
        else:
            all_iou.append(record[l3][-3])
    all_iou = np.array(all_iou, dtype=np.float)
    max_idx = np.argmax(all_iou)

    Data = np.load(os.path.join(path_npz, '101_data.npz'))
    seqs, denselabel = Data['data'], Data['denselabel']
    seqs = seqs.transpose((0, 2, 1))
    pos_seq = []
    for seq_tmp in open(path_npz + '/positive'):
        if seq_tmp[0] != '>':
            pos_seq.append(seq_tmp[:-1])
    ##
    cv_num = 5
    interval = int(len(seqs) / cv_num)
    index = range(len(seqs))
    # choose the 1-fold cross validation
    index_test = index[max_idx * interval:(max_idx + 1) * interval]
    index_train = list(set(index) - set(index_test))
    # build test data generator
    data_te = seqs[index_test]
    denselabel_te = denselabel[index_test]
    test_seq = []
    for sind in index_test:
        test_seq.append(pos_seq[sind].replace('T','U'))
    test_data = CRIDataSetTest(data_te, denselabel_te)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)
    # Load weights
    checkpoint_file = data_dir + '/101_HNML_0.7_FCN_model_best' + str(max_idx) + '.pth'
    chk = torch.load(checkpoint_file)
    state_dict = chk['model_state_dict']
    model = FCNP(motiflen=motifLen)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    #motif_wight=''
    denselabel_test = []
    score = []
    for i_batch, sample_batch in enumerate(test_loader):
        X_data = sample_batch["data"].float().to(device)
        with torch.no_grad():
            denselabel_p, score_p, conv_weight = model(X_data)
            denselabel_test.append(denselabel_p.data.cpu().numpy())
            score.append(score_p[0].data.cpu().numpy())
            motif_wight = conv_weight.data.cpu().numpy()
    score = np.array(score,dtype=np.float)
    score = score.transpose((0,2,1))
    print(score.shape)
    print(motif_wight.shape)
    filter_weight = []
    for x in motif_wight:
        x = x - x.mean(axis=0)
        filter_weight.append(x)
    filter_weight = np.array(filter_weight)
    if os.path.exists(store_path + name):
        os.remove(store_path + name)
    os.makedirs(store_path + name)
    sample_i = 0
    if index1 == 0:
        get_motif_fig_new(filter_weight, score, store_path + name, test_seq,sample_i)
    print('done')



if __name__ == "__main__":
    dpath = '/store_data/result_fca/kn128_64_ks12_rmsprop_fcnap/101'
    npz_path = '/home/szhen/Downloads/CPBPFCNA/all_data'
    store_path = '/home/szhen/Downloads/CPBPFCNA/motif_new/101/'
    seq_len = 101
    motiflen = 12
    gpu = '0'
    for pname in os.listdir(dpath):
        main(dpath + '/' + pname,npz_path + '/' + pname, pname, gpu, motiflen, store_path)
