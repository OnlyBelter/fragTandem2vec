import fasttext
import pandas as pd
import numpy as np
import pickle
import argparse
import os


def load_mol_frag_model(model_path):
    return fasttext.load_model(model_path)


def load_kmeans_model(kmenas_path):
    with open(kmenas_path, 'rb') as handle:
        return pickle.load(handle)


def get_mol_vec(frag_vec_model, frag):
    """
    get molecular vector by sum all fragment vectors
    :return:
    """
    frags = frag.split(',')
    frag2vec = []
    for f in frags:
        f_v = frag_vec_model.get_word_vector(f)
        frag2vec.append(f_v)
    frag2vec = np.array(frag2vec)
    # print(frag2vec)
    mol_vec = np.sum(frag2vec, axis=0)
    return mol_vec


def predict_class(kmeans_model, mol_vec_df, pure_kmeans_class):
    """
    predict class,
    :param kmeans_model:
    :param mol_vec_df: cid, mol_vec
    :param pure_kmeans_class: a dict, kmeans_class2classyFire_super_class
    :return:
    """
    cid2kmeans_class = {}
    for i in mol_vec_df.index:
        mol_vec = mol_vec_df.loc[i, :]
        cid2kmeans_class[i] = kmeans_model.predict(mol_vec.values.reshape(1, -1))[0]
    cid2kmeans_class_df = pd.DataFrame.from_dict(data=cid2kmeans_class, orient='index')
    # print(cid2kmeans_class_df.shape)
    cid2kmeans_class_df.rename(columns={0: 'kmeans_class'}, inplace=True)
    cid2kmeans_class_df['super_class_in_classyFire'] = 'mixed class'
    for i in cid2kmeans_class_df.index:
        kmeans_class = cid2kmeans_class_df.loc[i, 'kmeans_class']
        if kmeans_class in pure_kmeans_class:
            cid2kmeans_class_df.loc[i, 'super_class_in_classyFire'] = pure_kmeans_class[kmeans_class]
    return cid2kmeans_class_df


def cosine_dis(v1, v2):
    return np.dot(v1, v2)/(np.linalg.norm(v1) * np.linalg.norm(v2))


def cosine_dis2(df, vec):
    dot_product = np.dot(df, vec)
    norm_produt = np.linalg.norm(df, axis=1) * np.linalg.norm(vec)
    return np.divide(dot_product, norm_produt)


def find_nn(training_mol_vec_fp, query_mol_vec_df, top_n):
    """
    find top_n nearest neighbors in all training set (more than 10,000,000 molecules)
    :param training_mol_vec_fp: molecular vector of all training set
    :param query_mol_vec_df: a data frame of molecular vector as query item, index is cid
    :param top_n: top n nearest neighbors, max is 100
    :return:
    """
    # cid2dis_top = {}
    # cid2distance = {}
    query2cid_dis = {}
    query2cid_dis_top = {}
    query2nn = []
    query_len = query_mol_vec_df.shape[0]
    index2cid = {i: query_mol_vec_df.index[i] for i in range(query_len)}
    if top_n > 100:
        top_n = 100
    with open(training_mol_vec_fp, 'r') as handel:
        counter = 0
        for i in handel:
            current_line = i.split(',')
            cid = current_line[0]
            mol_vec = [float(v) for v in current_line[1:]]
            _cosine_dis = cosine_dis2(query_mol_vec_df, mol_vec)
            for j in range(query_len):
                q_cid = index2cid[j]
                # q_mol_vec = query_mol_vec_df.loc[q_cid, :]
                if q_cid not in query2cid_dis:
                    query2cid_dis[q_cid] = {}
                query2cid_dis[q_cid][cid] = _cosine_dis[j]
            # cid2distance[cid] = cosine_dis(q_mol_vec, mol_vec)
            if len(query2cid_dis[index2cid[0]]) >= 1000:
                for q_cid2 in query_mol_vec_df.index:
                    cid2distance_sorted = sorted(query2cid_dis[q_cid2].items(), key=lambda x: x[1], reverse=True)
                    # cid2distance_df = pd.DataFrame.from_dict(query2cid_dis[q_cid2], orient='index')
                    # cid2distance_df.sort_values(by=[0], inplace=True, ascending=False)
                    cid2distance_topn = cid2distance_sorted[:top_n].copy()
                    if q_cid2 not in query2cid_dis_top:
                        query2cid_dis_top[q_cid2] = {}
                    query2cid_dis_top[q_cid2].update({i[0]: i[1] for i in cid2distance_topn})
                    query2cid_dis[q_cid2] = {}
            if counter % 10000 == 0:
                print('current line: {}'.format(counter))
            if counter >= 100000:
                break
            counter += 1
    for q_cid in query_mol_vec_df.index:
        cid2distance_sorted = sorted(query2cid_dis[q_cid2].items(), key=lambda x: x[1], reverse=True)
        cid2distance_topn = cid2distance_sorted[:top_n].copy()
        query2cid_dis_top[q_cid].update({i[0]: i[1] for i in cid2distance_topn})
        top_dis_df = pd.DataFrame.from_dict(query2cid_dis_top[q_cid], orient='index')
        top_dis_df.sort_values(by=[0], inplace=True, ascending=False)
        top_n_dis = top_dis_df.iloc[range(top_n), :].copy().to_dict()[0]
        sorted_top_n_dis = sorted(top_n_dis.items(), key=lambda x: x[1], reverse=True)
        query2nn.append({q_cid: '; '.join(i[0] + ': ' + str(i[1]) for i in sorted_top_n_dis)})
    return query2nn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='molecule class prediction and finding nearest neighbors')
    parser.add_argument('frag_vec_model_fp', help='file path of trained fragment2vector model')
    parser.add_argument('input_fp', help='file path of cid2frag_id_sentence')
    parser.add_argument('--clustering_model_fp', help='file path of trained clustering model')
    parser.add_argument('--training_mol_vec_fp', default='', help='file path of molecular vectors of all training set, '
                                                                  'or a small part of training set where wants to find '
                                                                  'nearest neighbors.')
    parser.add_argument('--pure_kmeans_class_fp', default='', help='file path of pure kmeans class')
    parser.add_argument('--result_dir', help='result directory')
    parser.add_argument('--topn', default=10, help='find top N of nearest neighbors')
    parser.add_argument('--log_fn', help='log file name')
    parser.add_argument('--find_nearest_neighbors', action='store_true', default=False, help='if find nearest neighbors')
    parser.add_argument('--predict_class', action='store_true', default=False, help='if predict class')

    args = parser.parse_args()
    input_fp = args.input_fp
    frag_vec_model_fp = args.frag_vec_model_fp


    result_dir = args.result_dir
    result_fp_mol_vec = os.path.join(result_dir, 'step3_mol_vec.csv')
    result_fp_nn = os.path.join(result_dir, 'step3_mol_nearest_neighbors.txt')
    topn = args.topn
    need_find_nn = args.find_nearest_neighbors
    need_predict_class = args.predict_class
    log_fp = args.log_fn
    frag_raw_info = pd.read_csv(input_fp, index_col=0, sep='\t', header=None)
    # print(frag_raw_info.head())
    frag_raw_info = frag_raw_info.loc[:, [1]].copy()
    # print(frag_raw_info.head())

    # find molecular vectors
    print()
    print('>>>  Calculating molecular vectors...')
    print()
    # print('>>>')
    cid2mol_vec = {}
    frag_vec_model = load_mol_frag_model(frag_vec_model_fp)
    for i in frag_raw_info.index:
        _frag = str(frag_raw_info.loc[i, 1])
        mol_vec = get_mol_vec(frag_vec_model, _frag)
        cid2mol_vec[i] = mol_vec
    mol_vec_df = pd.DataFrame.from_dict(data=cid2mol_vec, orient='index')
    mol_vec_df.to_csv(result_fp_mol_vec)

    # find nearest neighbors
    if need_find_nn:
        print()
        print('>>>  Finding nearest neighbor of top {}...'.format(topn))
        print()
        # print('>>>')
        training_mol_vec_fp = args.training_mol_vec_fp
        if not training_mol_vec_fp:
            print('  Error: need to add the file path of "mol2vec_related.csv" by parameter --training_mol_vec_fp')
            raise FileNotFoundError
        with open(result_fp_nn, 'w') as handle:
            handle.write('\t'.join(['CID', 'nearest_neighbors']) + '\n')
        # for i in mol_vec_df.index:
        #     current_mol_vec = mol_vec_df.loc[i, :]
        query_nn = find_nn(training_mol_vec_fp, mol_vec_df, top_n=topn)
        for nn in query_nn:
            with open(result_fp_nn, 'a') as handle:
                handle.write('\t'.join([str(list(nn.keys())[0]), list(nn.values())[0]]) + '\n')

    # predict class of each molecule
    if need_predict_class:
        print()
        print('>>>  Predicting class of each molecule ...')
        print()
        # print('>>>')
        clustering_model_fp = args.clustering_model_fp
        pure_kmeans_class_fp = args.pure_kmeans_class_fp
        if not pure_kmeans_class_fp:
            print('need to add the file path of "pure_kmeans_class.csv" by parameter --pure_kmeans_class_fp')
            raise FileNotFoundError
        pure_kmeans_class = pd.read_csv(pure_kmeans_class_fp, index_col=0)
        pure_kemans_class2super_class = pure_kmeans_class.to_dict()['pure_superclass']
        kmeans_model = load_kmeans_model(clustering_model_fp)
        predicted_class = predict_class(kmeans_model=kmeans_model, mol_vec_df=mol_vec_df,
                                        pure_kmeans_class=pure_kemans_class2super_class)
        predicted_class.to_csv(os.path.join(result_dir, 'step3_predicted_class.csv'))
