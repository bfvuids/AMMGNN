import torch
import numpy as np
import argparse
import random
from itertools import combinations
from utils import load_citation, sgc_precompute, set_seed
from meta import Meta
from sgc_data_generator_multi import sgc_data_generator_multi_attr, label_select
from load_data_multi import load_data_multi
from normalization import fetch_normalization, row_normalize

def main(args):
    step = args.step
    set_seed(args.seed)
    adj, features, labels = load_data_multi(args.dataset)
    features = sgc_precompute(features, adj, args.degree)

    node_num = args.nodenum
    class_label = [x for x in range(args.classnum)] 
    combination = [list(random.sample(class_label, args.num))]
    
    label_dict = label_select(labels, node_num)
    config = [
        ('linear', [args.hidden, features.size(1)]),
        ('linear', [args.n_way, args.hidden])
    ]
    all_label = [[x for x in range(node_num)]]

    for i in range(len(combination)):
        print("Cross Validation: {}".format((i + 1)))
        maml = Meta(args, config, all_label)
        test_label = list(combination[i])
        valid_num = 10

        class_list_left = list(set(class_label).difference(set(test_label)))
        val_label = random.sample(class_list_left, valid_num)
        train_label = list(set(class_list_left).difference(set(val_label)))
        print('Cross Validation {} Train_Label_List: {} '.format(i + 1, train_label))
        print('Cross Validation {} Val_Label_List: {} '.format(i + 1, val_label))
        print('Cross Validation {} Test_Label_List: {} '.format(i + 1, test_label))
        max_val_acc = 0

        for j in range(args.epoch):
            
            x_spt, y_spt, x_qry, y_qry, y_idx, all_features = sgc_data_generator_multi_attr(features, labels, node_num, train_label, args.task_num, args.n_way, args.k_spt, args.k_qry, label_dict,features)
            accs = maml.forward_less(x_spt, y_spt, x_qry, y_qry, y_idx, all_features)
            print('Step:', j, '\tMeta_Training_Accuracy:', accs, [accs[-1]])
            
            if j % 100 == 0:
                torch.save(maml.state_dict(), 'maml.pkl')
                meta_test_acc_less = []
                meta_test_acc_less_new = []
                meta_val_acc = []
                meta_val_acc_new = []
    
                for k in range(10):
                    model_meta_val = Meta(args, config, all_label)
                    model_meta_val.load_state_dict(torch.load('maml.pkl'))
                    model_meta_val.eval()
                    x_spt, y_spt, x_qry, y_qry, y_idx, all_features = sgc_data_generator_multi_attr(features, labels, node_num, val_label, args.task_num, args.n_way, args.k_spt, args.k_qry, label_dict, features)
                    accs_val = model_meta_val.forward_less(x_spt, y_spt, x_qry, y_qry, y_idx, all_features)
                    meta_val_acc.append(accs_val)
                    meta_val_acc_new.append([accs_val[-1]])




if __name__ == '__main__':

    argparser = argparse.ArgumentParser()

    argparser.add_argument('--epoch', type=int, help='epoch number', default=999999)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.003)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.5)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=10)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=3)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=12)
    argparser.add_argument('--hidden', type=int, help='Number of hidden units', default=16)
    argparser.add_argument('--dataset', type=str, default='dblp', help='Dataset to use.')
    argparser.add_argument('--nodenum', type=int, default=1, help='node number.')
    argparser.add_argument('--classnum', type=int, default=1, help='class number.')
    argparser.add_argument('--num', type=int, default=1, help='.')
    argparser.add_argument('--normalization', type=str, default='AugNormAdj', help='Normalization method for the adjacency matrix.')
    argparser.add_argument('--seed', type=int, default=42, help='Random seed.')
    argparser.add_argument('--degree', type=int, default=2, help='degree of the approximation.')
    argparser.add_argument('--step', type=int, default=50, help='How many times to random select node to test')

    args = argparser.parse_args()

    main(args)
