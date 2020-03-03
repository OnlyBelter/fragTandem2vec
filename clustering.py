from sklearn.cluster import KMeans, DBSCAN
import pandas as pd
import pickle
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
import argparse
import os


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
    parser.add_argument('--include_small_dataset_dir', help='directory of included small dataset')
    parser.add_argument('--model', help='which model, molFrag2vec or mol2vec?')
    parser.add_argument('--cluster_method', default='dbscan', help='the method of clustering')
    parser.add_argument('--repeat', default=1, help='how many times to do clustering')
    args = parser.parse_args()
    input_dir = args.input_dir
    result_dir = args.result_dir
    include_small_dataset_dir = args.include_small_dataset_dir

    model_name = args.model
    cluster_method = args.cluster_method
    n_repeat = args.repeat
    cid2class_classyfire = pd.read_csv(os.path.join(include_small_dataset_dir,
                                                    'down_sampled_cid2class_unique.csv'), index_col='CID')

    # cid2class_classyfire = pd.read_csv('./big-data/cid2class_classyfire/down_sampled_cid2class_unique.csv', index_col='CID')

    # step 1: training clustering model
    mol2vec_fp = os.path.join(input_dir, 'step4_selected_mol2vec_model_{}.csv'.format(model_name))
    # mol2vec_fp = './big-data/vectors/mol2vec_model_{}.csv'.format(model_name)
    x = load_mol_vec(mol2vec_fp)
    for i in range(n_repeat):
        print('>>> {}'.format(i))
        model_path = os.path.join(result_dir,
                                  'step5_{}_model_{}_{}.pkl'.format(cluster_method, model_name, i))
        cid2class_fp = os.path.join(result_dir, 'step5_cid2class_model_{}_{}_{}.csv'.format(model_name, cluster_method, i))
        predicted_class = cluster_predict(mol_vec_x=x, n_clusters=100, model_path=model_path, cluster_model=cluster_method)
        x['predicted_class'] = predicted_class
        cid2class = cid2class_classyfire.merge(x, left_index=True, right_index=True)
        cid2class.to_csv(cid2class_fp, index_label='CID', columns=['superclass', 'predicted_class'])

    # step2: calculate purity score
    for i in range(n_repeat):
        print('>>> {}'.format(i))
        cid2class_fp = os.path.join(result_dir,
                                    'step5_cid2class_model_{}_{}_{}.csv'.format(model_name, cluster_method, i))
        result_fp = os.path.join(result_dir,
                                 'step5_cid2class_with_purity_score_model_{}_{}_{}.csv'.format(model_name, cluster_method, i))
        cid2class_with_purity_score = calculate_purity_score(cid2class_fp)
        cid2class_with_purity_score.sort_values(by=['purity_score', 'superclass_name'], inplace=True, ascending=False)
        cid2class_with_purity_score.to_csv(result_fp)

    # step3: comparing two models
    result_fp2 = os.path.join(result_dir, 'step5_superclass_in_pure_cluster_{}.csv'.format(model_name))
    compare_two_model(model_name=model_name, result_fp=result_fp2, n_repeat=n_repeat)
