import random
import torch

def label_select(labels, node_num):
    labels_local = labels.clone().detach().long()
    label_dict = {}
    for j in range(node_num):
        if labels_local[j].item() not in label_dict.keys():
            label_dict[labels_local[j].item()] = [j]
        else:
            label_dict[labels_local[j].item()].append(j)
    return label_dict
    
def sgc_data_generator_multi_attr(features, labels, node_num, select_array, task_num, n_way, k_spt, k_qry, label_dict, features_orig):
    x_spt = []
    y_spt = []
    x_qry = []
    y_qry = []
    y_qry_idx = []
    all_features = []
    for class_batch in range(1):
        all_index = []
        class_idx = []
        class_train = []
        class_test = []
        labels_local = labels.clone().detach()
        select_class = random.sample(select_array, n_way)
        for ways in range(n_way):
            class_train.append([])
            class_test.append([])
        for class_num in select_class:
            labels_local[label_dict[class_num]] = select_class.index(class_num)
            class_idx.append(label_dict[class_num])
        for t in range(task_num):
            for ways in range(n_way):
                class_train[ways] = random.sample(class_idx[ways], k_spt)
                class_test[ways] = [n1 for n1 in class_idx[ways] if n1 not in class_train[ways]]
                class_test[ways] = random.sample(class_test[ways], k_qry)
            train_idx = []
            for ways in range(n_way):
                train_idx = train_idx + class_train[ways]
            random.shuffle(train_idx)
            test_idx = []
            for ways in range(n_way):
                test_idx = test_idx + class_test[ways]
            random.shuffle(test_idx)
            x_spt.append(features[train_idx])
            y_spt.append(labels_local[train_idx].long())
            x_qry.append(features[test_idx])
            y_qry.append(labels_local[test_idx].long())
            y_qry_idx.append(test_idx)
        for ways in range(n_way):
            all_index += class_idx[ways]
        random.shuffle(all_index)
        all_features.append(features_orig[all_index])
    return x_spt, y_spt, x_qry, y_qry, y_qry_idx, all_features