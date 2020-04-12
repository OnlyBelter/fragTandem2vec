"""
best selected parameters: ws:4, minn: 1, maxn: 2
and parallel without refragment is better than parallel with refragment
# usage
$ python pipeline1_train_frag_vec2vis.py big-data/moses_dataset/result/step2_parallel_frag_smiles_sentence.csv
big-data/moses_dataset/parallel_frag_info.csv /mnt/data/frag2vec_test/
"""

import os
import time
import argparse
import pandas as pd
from pub_func import get_format_time
from training_frag_vec_model import train_model, get_frag_vector
from sklearn.manifold import TSNE
from helper_func import SELECTED_MD

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns


def reduce_by_tsne(x):
    t0 = time.time()
    tsne = TSNE(n_components=2, n_jobs=4, learning_rate=200,
                n_iter=2000, random_state=42, init='pca', verbose=1)
    X_reduced_tsne = tsne.fit_transform(x)
    # X_reduced_tsne = tsne.fit(x)
    print(X_reduced_tsne.shape)
    # np.save('X_reduced_tsne_pca_first', X_reduced_tsne2)
    t1 = time.time()
    print("t-SNE took {:.1f}s.".format(t1 - t0))
    return X_reduced_tsne


def show_each_md(x_reduced, frag_info, file_path=''):
    """
    reduced_x: 2 dimensions x with fragment as index, a dataframe
    frag_info: the number of each MD with fragemnt as index, a dataframe
    """
    # model = model_name
    fig, ax = plt.subplots(2, 4, figsize=(24, 12))
    ax = ax.flatten()
    # print(x_reduced.head(2))
    # print(frag_info.head(2))
    intersect_index = set(x_reduced.index.to_list()) & set(frag_info.index.to_list())
    x_reduced = x_reduced.loc[intersect_index, :].copy()  # alignment
    frag_info = frag_info.loc[intersect_index, :].copy()
    # reduced_x = reduced_x.loc[frag_info.index, :].copy()
    # parallel_frag_info = parallel_frag_info.loc[:, selected_md].copy()
    for i,md in enumerate(frag_info.columns.to_list()):
        # current_labels = parallel_frag_info.iloc[:, i]
        current_labels = frag_info.iloc[:, i]
        unique_labels = sorted(current_labels.unique())
        n_labels = len(unique_labels)
        # print(n_labels)
        cc = sns.color_palette('Blues', n_labels)
        for j,label in enumerate(unique_labels):
            current_nodes = (current_labels == label)
            ax[i].scatter(x_reduced.loc[current_nodes, 0], x_reduced.loc[current_nodes, 1],
                          c=colors.rgb2hex(cc[j]), vmin=0, vmax=10, s=10, label=str(label))
        ax[i].set_title(md, fontsize=12)
        ax[i].legend()
    plt.tight_layout()
    plt.savefig(file_path, bbox_inches='tight', transparent=True)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Training molFrag2vec model using FastText')
    parser.add_argument('input_fn', help='the file path of training set, frag_smiles_sentence')
    parser.add_argument('frag_info_fn', help='the file path of fragment infomation '
                                             'which contains the numbor of selected MD')
    parser.add_argument('result_dir', help='the directory of result files')
    # parser.add_argument('--model_fn', help='file name of trained model')
    # parser.add_argument('--frag_vec_fn', help='file name of fragment vector')
    args = parser.parse_args()
    input_file = args.input_fn
    frag_info = args.frag_info_fn
    result_dir = args.result_dir
    model_fn = 'model_ws_{}_minn_{}_maxn_{}.bin'
    frag_vec_fn = 'frag2vec_ws_{}_minn_{}_maxn_{}.csv'

    t0 = get_format_time()
    # m_nn_set = set([(i, i*j) for i in range(1, 2) for j in range(4, 5)])
    m_nn_set = {(1, 2)}
    # m_nn_set = set([(i, j) for i in range(1,3) for j in range(2, 5)])
    ws_list = [4]
    for ws in ws_list:
        for m_nn in m_nn_set:
            ws = ws
            minn = m_nn[0]
            maxn = m_nn[1]
            epoch = 30
            # epoch = 50
            print('  >Start to train vector model in {}, with ws={}, minn={}, maxn={}...'.format(t0, ws, minn, maxn))
            sub_dir = 'sub_ws_{}_minn_{}_maxn_{}'.format(ws, minn, maxn)
            if not os.path.exists(os.path.join(result_dir, sub_dir)):
                os.mkdir(os.path.join(result_dir, sub_dir))
            model_fp = os.path.join(result_dir, sub_dir, model_fn.format(ws, minn, maxn))
            frag2vec_fp = os.path.join(result_dir, sub_dir, frag_vec_fn.format(ws, minn, maxn))
            train_model(input_file, model_fp, ws=ws, minn=minn, maxn=maxn)
            # mol2vec_fp = os.path.join(result_dir, 'selected_mol2vec.csv')
            get_frag_vector(model_fp, frag_id2vec_fp=frag2vec_fp)
            t1 = get_format_time()
            print('  >Finished training vector model in {}...'.format(t1))

            print('  >Start to reduce fragment vector by t-SNE...')
            frag2info = pd.read_csv(frag_info, index_col='fragment')
            frag2info = frag2info.loc[:, SELECTED_MD].copy()
            frag2vec = pd.read_csv(frag2vec_fp, index_col='fragment')
            x_reduced = reduce_by_tsne(frag2vec)
            x_reduced = pd.DataFrame(data=x_reduced, index=frag2vec.index)
            save_fig_path = os.path.join(result_dir, sub_dir, 't-SNE_vis_ws_{}_minn_{}_maxn_{}.pdf'.format(ws, minn, maxn))

            print('  >Start to plot t-SNE vis of fragment vector...')
            show_each_md(x_reduced=x_reduced, frag_info=frag2info,
                         file_path=save_fig_path)
