#!/usr/bin/python

import os
import sys
import time
import argparse
import math
import numpy as np
import os.path as osp

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# custom functions defined by user
from FCNmotifP import FCN, FCNA
from datasets import CRIDataSetTrain, CRIDataSetTest
from trainer import Trainer
from loss import OhemNegLoss
from utils import Dict


def get_args():
    """Parse all the arguments.

        Returns:
          A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Fully Convolutional Neural Network for CircRNA RBP Interaction Sites Identification")### some parms need change

    parser.add_argument("-d", dest="data_dir", type=str, default='/home/szhen/Downloads/circ-protein/code-data/all_large',
                        help="A directory containing the training data.")
    parser.add_argument("-ld", dest="data_length", type=str, default='101',
                        help="seq data length.")
    parser.add_argument("-g", dest="gpu", type=str, default='0',
                        help="choose gpu device. eg. '0,1,2' ")
    parser.add_argument("-s", dest="seed", type=int, default=5,
                        help="Random seed to have reproducible results.")
    # Arguments for Adam or SGD optimization
    parser.add_argument("-b", dest="batch_size", type=int, default=200,
                        help="Number of sequences sent to the network in one step.")
    parser.add_argument("-lr", dest="learning_rate", type=float, default=0.01,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("-t", dest="threshold", type=float, default=0.5,
                        help="Threshold value for positive")
    parser.add_argument("-e", dest="max_epoch", type=int, default=40,
                        help="Number of training steps.")
    parser.add_argument("-w", dest="weight_decay", type=float, default=0.0005,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("-r", dest="restore", type=str, default=None,
                        help="Where restore model parameters from.")
    parser.add_argument("-ms", dest="model_select", type=str, default=None,
                        help="confirm model type")
    parser.add_argument("-ls", dest="loss_function", type=str, default=None,
                        help="loss function")
    parser.add_argument("-lt", dest="loss_threshold", type=str, default=None,
                        help="HNML threshold")

    return parser.parse_args()

def run_model(dname):
    args = get_args()
    begin_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        if len(args.gpu.split(',')) == 1:
            device = torch.device("cuda:" + args.gpu)
        else:
            device = torch.device("cuda:" + args.gpu.split(',')[0])
    else:
        device = torch.device("cpu")
        torch.manual_seed(args.seed)
    if args.data_length == '101':
        Data = np.load(osp.join(args.data_dir + '/' + dname, '%s_data.npz' % args.data_length),allow_pickle=True)  ### need change
    elif args.data_length == '201':
        Data = np.load(osp.join(args.data_dir + '/' + dname, '%s_data_new.npz' % args.data_length),allow_pickle=True)  ### need change
    elif args.data_length == '501':
        Data = np.load(osp.join(args.data_dir + '/' + dname, '%s_data_new.npz' % args.data_length),allow_pickle=True)  ### need change
    print(args.data_dir + '/' + dname, '%s_data.npz' % args.data_length)
    try:
        seqs, denselabel = Data['data'], Data['denselabel']
        seqs = seqs.transpose((0, 2, 1))
    except:
        print('can not load data:', dname)
        ##
    cv_num = 5
    interval = int(len(seqs) / cv_num)
    index = range(len(seqs))
    print(args.model_select)
    if args.loss_function == 'BCE':
        print('create record file')
        f = open(osp.join(args.data_dir + '/' + dname, args.data_length + '_' + args.loss_function + '_' + args.model_select + '_record.txt'), 'w')
        f.write('CV\tTrial\tIOU\tIOU_0\tIOU_1\n')
    elif args.loss_function == 'HNML':
        print('create record file')
        f = open(osp.join(args.data_dir + '/' + dname, args.data_length + '_' + args.loss_function + '_' + args.loss_threshold + '_' + args.model_select + '_record.txt'), 'w')
        f.write('CV\tTrial\tIOU\tIOU_0\tIOU_1\n')
    iou_mean = 0.
    iou_0_mean = 0
    iou_1_mean = 0
    for cv in range(cv_num):
        index_test = index[cv * interval:(cv + 1) * interval]
        index_train = list(set(index) - set(index_test))
        # build training data generator
        data_tr = seqs[index_train]
        denselabel_tr = denselabel[index_train]
        train_data = CRIDataSetTrain(data_tr, denselabel_tr)  ### need change
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                      num_workers=8)
        # build test data generator
        data_te = seqs[index_test]
        denselabel_te = denselabel[index_test]
        test_data = CRIDataSetTest(data_te, denselabel_te)  ### need change
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=8)
            # we implement many trials for different weight initialization
        iou_best = 0
        iou_0_best = 0
        iou_1_best = 0
        trial_best = 0
        for trial in range(5):
            if args.model_select == 'FCN':
                model = FCN(motiflen=12)
            elif args.model_select == 'FCNA':
                model = FCNA(motiflen=12)
            optimizer = optim.RMSprop(model.parameters(),
                                       lr=args.learning_rate,
                                       weight_decay=args.weight_decay)
            if args.loss_function == 'BCE':
                criterion = nn.BCELoss()
            elif args.loss_function == 'HNML':
                criterion = OhemNegLoss(device,float(args.loss_threshold))
            #criterion = nn.BCELoss()
            start_epoch = 0
            if args.restore:
                print("Resume it from {}.".format(args.restore_from))
                checkpoint = torch.load(args.restore)
                state_dict = checkpoint["model_state_dict"]
                model.load_state_dict(state_dict)

            # if there exists multiple GPUs, using DataParallel
            if len(args.gpu.split(',')) > 1 and (torch.cuda.device_count() > 1):
                model = nn.DataParallel(model, device_ids=[int(id_) for id_ in args.gpu.split(',')])

            executor = Trainer(model=model,
                                   optimizer=optimizer,
                                   criterion=criterion,
                                   device=device,
                                   start_epoch=start_epoch,
                                   max_epoch=args.max_epoch,
                                   train_loader=train_loader,
                                   test_loader=test_loader,
                                   thres=args.threshold)
            iou, iou_0, iou_1, state_dict = executor.train()
            if iou_best < iou :
                iou_best = iou
                iou_0_best = iou_0
                iou_1_best = iou_1
                trial_best = trial
                if args.loss_function == 'BCE':
                    checkpoint_file = osp.join(args.data_dir + '/' + dname, args.data_length + '_' + args.loss_function + '_' + args.model_select + '_model_best%d.pth' % cv)
                    torch.save({'model_state_dict': state_dict}, checkpoint_file)
                elif args.loss_function == 'HNML':
                    checkpoint_file = osp.join(args.data_dir + '/' + dname, args.data_length + '_' + args.loss_function + '_' + args.loss_threshold + '_' + args.model_select + '_model_best%d.pth' % cv)
                    torch.save({'model_state_dict': state_dict}, checkpoint_file)
        iou_mean += iou_best
        iou_0_mean += iou_0_best
        iou_1_mean += iou_1_best

        f.write("{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(cv, trial_best, iou_best, iou_0_best, iou_1_best))
        f.flush()
    end_time = time.time()

    f.write("mean\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(iou_mean / cv_num, iou_0_mean / cv_num, iou_1_mean / cv_num, end_time - begin_time))
    f.close()

def main():
    """Create the model and start the training."""
    args = get_args()
    ### load target sites length data, may use or not, later implement###
    #motifLen_dict = Dict(os.getcwd() + '/motifLen.txt')### need change
    #motifLen = motifLen_dict[args.name]### need change
    for dname in os.listdir(args.data_dir):
        if args.loss_function == 'BCE':
            record_path = osp.join(args.data_dir + '/' + dname, args.data_length + '_' + args.loss_function + '_' + args.model_select + '_record.txt')
            if os.path.exists(record_path):
                record_data = []
                for recon in open(record_path):
                    record_data.append(recon)
                if len(record_data) == 7:
                    continue
                elif len(record_data) == 0:
                    print('delete record file', record_path)
                    os.remove(record_path)
                    model_path = osp.join(args.data_dir + '/' + dname, args.data_length + '_' + args.loss_function + '_' + args.model_select + '_model_best0.pth')
                    print('delete best model',model_path)
                    try:
                        os.remove(model_path)
                    except:
                        print('no best model found',model_path)
                elif len(record_data) == 1:
                    print('delete record file', record_path)
                    os.remove(record_path)
                    model_path = osp.join(args.data_dir + '/' + dname, args.data_length + '_' + args.loss_function + '_' + args.model_select + '_model_best0.pth')
                    print('delete best model',model_path)
                    try:
                        os.remove(model_path)
                    except:
                        print('no best model found',model_path)
                elif len(record_data) > 1 and len(record_data) < 7:
                    re_len = len(record_data)
                    model_num = re_len - 1
                    print('delete record file', record_path)
                    os.remove(record_path)
                    for imodel in range(model_num):
                        model_path = osp.join(args.data_dir + '/' + dname, args.data_length + '_' + args.loss_function + '_' + args.model_select + '_model_best%d.pth' % imodel)
                        print('delete best model', model_path)
                        os.remove(model_path)
                    try:
                        os.remove(osp.join(args.data_dir + '/' + dname, args.data_length + '_' + args.loss_function + '_' + args.model_select + '_model_best%d.pth' % model_num))
                    except:
                        print('no best model found',osp.join(args.data_dir + '/' + dname, args.data_length + '_' + args.loss_function + '_' + args.model_select + '_model_best%d.pth' % model_num))
                run_model(dname)
            else:
                run_model(dname)
        elif args.loss_function == 'HNML':
            record_path = osp.join(args.data_dir + '/' + dname, args.data_length + '_' + args.loss_function + '_' + args.loss_threshold + '_' + args.model_select + '_record.txt')
            if os.path.exists(record_path):
                record_data = []
                for recon in open(record_path):
                    record_data.append(recon)
                if len(record_data) == 7:
                    continue
                elif len(record_data) == 0:
                    print('delete record file', record_path)
                    os.remove(record_path)
                    model_path = osp.join(args.data_dir + '/' + dname, args.data_length + '_' + args.loss_function + '_' + args.loss_threshold + '_' + args.model_select + '_model_best0.pth')
                    print('delete best model',model_path)
                    try:
                        os.remove(model_path)
                    except:
                        print('no best model found',model_path)
                elif len(record_data) == 1:
                    print('delete record file', record_path)
                    os.remove(record_path)
                    model_path = osp.join(args.data_dir + '/' + dname, args.data_length + '_' + args.loss_function + '_' + args.loss_threshold + '_' + args.model_select + '_model_best0.pth')
                    print('delete best model',model_path)
                    try:
                        os.remove(model_path)
                    except:
                        print('no best model found',model_path)
                elif len(record_data) > 1 and len(record_data) < 7:
                    re_len = len(record_data)
                    model_num = re_len - 1
                    print('delete record file', record_path)
                    os.remove(record_path)
                    for imodel in range(model_num):
                        model_path = osp.join(args.data_dir + '/' + dname, args.data_length + '_' + args.loss_function + '_' + args.loss_threshold + '_' + args.model_select + '_model_best%d.pth' % imodel)
                        print('delete best model', model_path)
                        os.remove(model_path)
                    try:
                        os.remove(osp.join(args.data_dir + '/' + dname, args.data_length + '_' + args.loss_function + '_' + args.loss_threshold + '_' + args.model_select + '_model_best%d.pth' % model_num))
                    except:
                        print('no best model found',osp.join(args.data_dir + '/' + dname, args.data_length + '_' + args.loss_function + '_' + args.loss_threshold + '_' + args.model_select + '_model_best%d.pth' % model_num))
                run_model(dname)
            else:
                run_model(dname)


if __name__ == "__main__":
    main()
