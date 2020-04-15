import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import normalize
from sklearn.mixture import BayesianGaussianMixture


def load_mol_vec(cid2vec_fp):
    """
    load molecular vectors
    :param cid2vec_fp:
    :return:
    """
    cid2vec = pd.read_csv(cid2vec_fp, index_col=0)
    return cid2vec


def cluster_predict(mol_vec_x, n_clusters, model_path, n_jobs=4, cluster_model='kmeans'):
    """
    clustering by KMeans
    :param mol_vec_x:
    :param n_clusters:
    :param model_path:
    :param result_path:
    :param n_jobs:
    :return:
    """
    if cluster_model == 'kmeans':
        _model = KMeans(n_clusters=n_clusters, n_jobs=n_jobs).fit(mol_vec_x)
        with open(model_path, 'wb') as f_handle:
            pickle.dump(_model, f_handle)
    if cluster_model == 'dbscan':
        _model = DBSCAN(eps=0.01, metric='cosine', min_samples=10, n_jobs=n_jobs).fit(mol_vec_x)
        with open(model_path, 'wb') as f_handle:
            pickle.dump(_model, f_handle)
    if cluster_model == 'bg':
        _model = BayesianGaussianMixture(weight_concentration_prior_type="dirichlet_process",
        n_components=30, reg_covar=0, max_iter=1500, mean_precision_prior=.8).fit(mol_vec_x)
        with open(model_path, 'wb') as f_handle:
            pickle.dump(_model, f_handle)
        return _model.predict(mol_vec_x)
    return _model.labels_


def calculate_purity_score(cid2predicted_class_fp):
    """

    :param cid2predicted_class_fp:
    :return:
    """
    cid2class = pd.read_csv(cid2predicted_class_fp)
    # groupby by predicted_class
    cid2class_groupby = cid2class.groupby('predicted_class')
    cid2class_groupby_count = cid2class_groupby.count()['CID']
    # cid2class_groupby_count.head()
    cluster2superclass = {}
    cluster2max_superclass = {}
    for name, group in cid2class_groupby:
        cluster2superclass[name] = group.groupby('superclass').count()
        sorted_cluster2superclass = cluster2superclass[name].sort_values(by='CID', ascending=False)
        cluster2max_superclass[name] = {'max': sorted_cluster2superclass.iloc[0, 1],
                                        'superclass_name': sorted_cluster2superclass.index[0],
                                        'total': sorted_cluster2superclass['CID'].sum()}
    cluster2max_superclass_df = pd.DataFrame.from_dict(cluster2max_superclass, orient='index')
    cluster2max_superclass_df['purity_score'] = cluster2max_superclass_df['max'] / cluster2max_superclass_df['total']
    return cluster2max_superclass_df


def compare_two_model(model_name, result_fp, n_repeat=10):
    cid2ps_list = []
    for i in range(n_repeat):
        cid2ps_fp = './big-data/purity_score/{}/cid2class_with_purity_score_{}.csv'.format(model_name, i)
        cid2ps = pd.read_csv(cid2ps_fp)
        cid2ps['batch'] = i
        cid2ps_list.append(cid2ps)
    cid2ps_all = pd.concat(cid2ps_list)
    print(cid2ps_all.shape)
    pure_cluster = cid2ps_all[cid2ps_all['purity_score'] >= 0.7].copy()
    print(pure_cluster.shape)
    total_pure_cluster_count = pure_cluster.groupby(['batch'])['total'].sum()
    total_pure_cluster_count_mean = np.mean(total_pure_cluster_count)
    total_pure_cluster_count_std = np.std(total_pure_cluster_count)
    print('>>> total_pure_cluster_count: {} +/- {}'.format(total_pure_cluster_count_mean, total_pure_cluster_count_std))
    superclass_in_pure_cluster = pure_cluster.groupby(['batch', 'superclass_name']).count()
    superclass_in_pure_cluster.to_csv(result_fp)

    pure_score = pure_cluster.groupby(['batch'])['purity_score'].mean()
    pure_score_mean = np.mean(pure_score)
    pure_score_std = np.std(pure_score)
    print('>>> pure score: {} +/- {}'.format(pure_score_mean, pure_score_std))


def cluster_by_dbscan(mol2vec, n_jobs=4, metric='euclidean', eps=0.5, min_samples=20):
    x = normalize(mol2vec, norm='l2')
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, n_jobs=n_jobs)
    cluster_labels = clustering.fit_predict(x)

    n_clusters_ = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise_ = list(cluster_labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    return cluster_labels


def cluster_by_gradient_dbscan(mol2vec, eps_range=(0.05, 0.35), step=0.05, max_min_samples=20):
    """
    change eps from 0.35 to 0.05 with 0.05 step size
    :param mol2vec:
    :param eps_range:
    :return:
    """
    max_threshold = 5000
    min_threshold = 50
    cid2label = {}
    e_start = int(eps_range[0] * 100)
    e_end = int(eps_range[1] * 100)
    step = int(step * 100)
    max_min_samples = max_min_samples

    for i,eps in enumerate(range(e_start, e_end + 1, step)):
        cid_need_recluster = []
        eps = eps * 0.01
        min_samples = max_min_samples - i
        if min_samples < 5:
            min_samples = 5
        print('# eps is {}, min_samples is {}'.format(eps, min_samples))

        cluster_labels = cluster_by_dbscan(mol2vec, eps=eps, min_samples=min_samples)
        _cid2labels = pd.DataFrame(data=cluster_labels, index=mol2vec.index, columns=['labels'])
        noise = _cid2labels[_cid2labels['labels'] == -1].index.to_list()
        for noi in noise:
            cid2label[noi] = 'noise'
        _cid2labels = _cid2labels[_cid2labels['labels'] != -1].copy()
        label2num = _cid2labels['labels'].value_counts().to_dict()
        print('  > current label2num: {}'.format(label2num))
        for lab, n in label2num.items():
            current_label_name = 'round_{}_'.format(i) + str(lab)
            current_cids = _cid2labels[_cid2labels['labels'] == lab].index.to_list()
            for _cid in current_cids:
                cid2label[_cid] = current_label_name
            if n > max_threshold:
                cid_need_recluster += current_cids
            elif n < min_threshold:
                cid_need_recluster += current_cids
        if cid_need_recluster or noise:
            mol2vec = mol2vec[mol2vec.index.isin(cid_need_recluster + noise)].copy()
        else:
            break
    cid2label_df = pd.DataFrame.from_dict(data=cid2label, orient='index', columns=['labels'])
    return cid2label_df


def cluster_analysis(md_info, cluster_labels, threshold=0.7, max_top_n = 3):
    """
    calculate purity score of each cluster and n_node, class_distribution
    :para md_info: a dataFrame contains the information of molecular descriptors, index(cid)/num_md
    :para cluster_labels: a dataFrame contains the label from DBSCAN algo, index(cid)/labels
    :para threshold: threshold for high-quality cluster based on purity_score
    """
    clusters = cluster_labels.loc[:, 'labels'].unique()
    cluster_info = pd.DataFrame(index=list(clusters), columns=['n_node', 'class_count', 'purity_score_of_each_md', 'n_class'])
    # cluster_info = {c: {'n_node': 0, 'class_count': {}, 'purity_score_of_each_md': [], 'n_class': 0} for c
    #                 in clusters}
    md_info[md_info >= 1] = 1
    md_label = md_info.apply(lambda x: ''.join([str(i) for i in x]), axis=1)
    md_label = pd.DataFrame(md_label, columns=['md_class'])
    n_clustered_node = 0  # the number of all nodes in high-quality clusters
    purity_score_in_hc = []  # purity score in high-quality clusters
    for c in clusters:
        # print('Current cluster is: {}'.format(c))
        current_nodes = cluster_labels[cluster_labels['labels'] == c].index
        current_md_label = md_label.loc[current_nodes, ['md_class']].copy()
        current_md_info = md_info.loc[current_nodes, :].copy()
        n_node = len(current_nodes)
        # print(current_md_label.head(2))
        class_value_counts = current_md_label['md_class'].value_counts().sort_values(ascending=False)
        class_count = class_value_counts.to_dict()  # count each class
        top_n = min(max_top_n, len(class_count))
        purity_score_of_topn = np.sum(class_value_counts.to_list()[:top_n]) / n_node
        purity_score_of_each_md = np.array(current_md_info.sum() / n_node)  # percentage of each md in current cluster

        cluster_info.loc[c, 'n_node'] = n_node  # number of nodes in this cluster
        cluster_info.loc[c, 'class_count'] = json.dumps(class_count)
        cluster_info.loc[c, 'n_class'] = len(class_count)
        cluster_info.loc[c, 'purity_score_of_each_md'] = ','.join(['{:.2f}'.format(i) for i in purity_score_of_each_md])
        cluster_info.loc[c, 'purity_score_of_topn'] = purity_score_of_topn
        if np.any(purity_score_of_each_md >= threshold):
            n_clustered_node += n_node
    return {'cluster_info': cluster_info, 'n_clustered_node': n_clustered_node}


def ks_test(data1, data2):
    """

    :param data1:
    :param data2:
    :return: p-value of ks test
    """
    ks_result = ks_2samp(data1, data2)
    return np.float('{:.2f}'.format(ks_result[1]))


def mol_sampling(mol2info, n_sample, n_repeat=100):
    """
    sampling molecules randomly
    :param mol2info:
    :param n_sample:
    :param n_repeat:
    :return: mean of MD percentage in sampled molecules and repeat 100 times
    """
    mean_md_percentage = []
    for i in range(n_repeat):
        sampled_mol = mol2info.sample(n=n_sample, random_state=42)
        _mean_md_per = sampled_mol.sum() / sampled_mol.shape[0]
        mean_md_percentage.append(_mean_md_per)
    return np.array(mean_md_percentage).mean(axis=0)


if __name__ == '__main__':
    # result_dir = './demo_data'
    # download_big_data_dir = './big-data'
    # include_small_dataset_dir = './dataset'
    # result_fp = os.path.join(result_dir, 'step4_selected_cid2fragment_down_sampled_model_mol2vec.csv')
    # result_fp2 = os.path.join(result_dir, 'step4_selected_cid2fragment_down_sampled_model_fragTandem2vec.csv')
    # selected_cid_fp = os.path.join(include_small_dataset_dir, 'down_sampled_cid2class_unique.csv')

    parser = argparse.ArgumentParser(
        description='Training molFrag2vec model using FastText')
    parser.add_argument('input_dir',
                        help='the directory of molecular vector files')
    parser.add_argument('result_dir',
                        help='where to save trained model')
    # parser.add_argument('--include_small_dataset_dir', help='directory of included small dataset')
    # parser.add_argument('--model', help='which model, molFrag2vec or mol2vec?')
    # parser.add_argument('--cluster_method', default='dbscan', help='the method of clustering')
    # parser.add_argument('--repeat', default=1, help='how many times to do clustering')
    args = parser.parse_args()
    input_dir = args.input_dir
    result_dir = args.result_dir
    # include_small_dataset_dir = args.include_small_dataset_dir

    # model_name = args.model
    # cluster_method = args.cluster_method
    # n_repeat = args.repeat
    # cid2class_classyfire = pd.read_csv(os.path.join(include_small_dataset_dir,
    #                                                 'down_sampled_cid2class_unique.csv'), index_col='CID')

    # cid2class_classyfire = pd.read_csv('./big-data/cid2class_classyfire/down_sampled_cid2class_unique.csv', index_col='CID')

    # step 1: training clustering model
    print('  > Start to clustering...')
    mol2vec_fp = os.path.join(input_dir, 'x_reduced_95%_PCA_parallel_without_refragment_mol2vec.csv')
    mol2info_fp = os.path.join(input_dir, 'mol2md_downsampled.csv')
    mol2vec = pd.read_csv(mol2vec_fp, index_col=0)
    print(mol2vec.shape)
    cid2class_fp = os.path.join(result_dir, 'step5_cid2class.csv')
    cid2class = cluster_by_gradient_dbscan(mol2vec=mol2vec, eps_range=(0.05, 0.25),
                                           step=0.01, max_min_samples=30)
    cid2class.to_csv(cid2class_fp)

    # step 2 cluster analysis
    cluster_info_fp = os.path.join(result_dir, 'step5_cluster_info.csv')
    mol2info = pd.read_csv(mol2info_fp, index_col='cid')
    cid2class = pd.read_csv(cid2class_fp, index_col=0)
    print('  > Start to analyze clusters...')
    ana_result = cluster_analysis(md_info=mol2info, cluster_labels=cid2class)
    print(ana_result['n_clustered_node'])
    cluster_info = ana_result['cluster_info']
    print('  > The shape of cluster_info: {}'.format(cluster_info.shape))
    # test MD distribution with random selected molecules

    # print('  > Start to calculate p-value...')
    # mol2info[mol2info >= 1] = 1
    # cluster_info['p-value'] = 1
    # for cluster in cluster_info.index:
    #     n_node = cluster_info.loc[cluster, 'n_node']
    #     purity_score_of_each_md = cluster_info.loc[cluster, 'purity_score_of_each_md'].split(',')
    #     data1 = np.array([np.float(i) for i in purity_score_of_each_md])
    #     data2 = mol_sampling(mol2info, n_sample=n_node)
    #     cluster_info.loc[cluster, 'p-value'] = ks_test(data1, data2)

    cluster_info.to_csv(cluster_info_fp, sep='\t')

    # for i in range(n_repeat):
    #     print('>>> {}'.format(i))
    #     model_path = os.path.join(result_dir,
    #                               'step5_{}_model_{}_{}.pkl'.format(cluster_method, model_name, i))
    #     cid2class_fp = os.path.join(result_dir, 'step5_cid2class_model_{}_{}_{}.csv'.format(model_name, cluster_method, i))
    #     predicted_class = cluster_predict(mol_vec_x=x, n_clusters=100, model_path=model_path, cluster_model=cluster_method)
    #     x['predicted_class'] = predicted_class
    #     cid2class = cid2class_classyfire.merge(x, left_index=True, right_index=True)
    #     cid2class.to_csv(cid2class_fp, index_label='CID', columns=['superclass', 'predicted_class'])

    # # step2: calculate purity score
    # for i in range(n_repeat):
    #     print('>>> {}'.format(i))
    #     cid2class_fp = os.path.join(result_dir,
    #                                 'step5_cid2class_model_{}_{}_{}.csv'.format(model_name, cluster_method, i))
    #     result_fp = os.path.join(result_dir,
    #                              'step5_cid2class_with_purity_score_model_{}_{}_{}.csv'.format(model_name, cluster_method, i))
    #     cid2class_with_purity_score = calculate_purity_score(cid2class_fp)
    #     cid2class_with_purity_score.sort_values(by=['purity_score', 'superclass_name'], inplace=True, ascending=False)
    #     cid2class_with_purity_score.to_csv(result_fp)
    #
    # # step3: comparing two models
    # result_fp2 = os.path.join(result_dir, 'step5_superclass_in_pure_cluster_{}.csv'.format(model_name))
    # compare_two_model(model_name=model_name, result_fp=result_fp2, n_repeat=n_repeat)
