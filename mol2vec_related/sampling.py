import numpy as np
import pandas as pd


def sampling_from_train_set(n_sample):
    total_n = 10530369
    return np.random.choice(range(total_n), n_sample, replace=False)

def get_selected_mol(cid2smiles_path, sentences_path, sampled_id, cid_result_path, sentence_result_path):
    """

    :param cid2smiles_path:
    :param sentences_path: molecular sentence formed by fragments (created by Mol2vec)
    :param sampled_id: a list which contains sampled id from range [0, total_n]
    :param cid_result_path: the file path of selected cid
    :param sentence_result_path: the file path of selected molecular sentence
    :return:
    """
    counter = 0
    with open(cid2smiles_path, 'r') as f_handle:
        for each_line in f_handle:
            cid, smiles = each_line.strip().split('\t')
            if counter in sampled_id:
                with open(cid_result_path, 'a') as f_handle2:
                    f_handle2.write('\t'.join([str(counter), cid]) + '\n')
            counter += 1
    counter2 = 0
    with open(sentences_path, 'r') as f_handle:
        for each_line in f_handle:
            each_line = each_line.strip()
            if counter2 in sampled_id:
                with open(sentence_result_path, 'a') as f_handle2:
                    f_handle2.write('\t'.join([str(counter2), each_line]) + '\n')
            counter2 += 1

if __name__ == '__main__':
    sampled_id = sampling_from_train_set(100000)
    cid2smiles_path = '../big-data/cid2fragment/mol2vec/cid2smiles_training_set.txt'
    sentences_path = '../big-data/cid2fragment/mol2vec/cid2smiles_training_set_coupus.txt'
    reuslt_path1 = '../big-data/cid2class_classyfire/mol2vec_related/selected_cid.tmp'
    reuslt_path2 = '../big-data/cid2class_classyfire/mol2vec_related/selected_sentences.tmp'
    reuslt_path3 = '../big-data/cid2class_classyfire/mol2vec_related/selected_cid2sentences.txt'
    # get_selected_mol(cid2smiles_path=cid2smiles_path, sentences_path=sentences_path, sampled_id=sampled_id,
    #                  cid_result_path=reuslt_path1, sentence_result_path=reuslt_path2)
    selected_cid = pd.read_csv(reuslt_path1, sep='\t', header=None, index_col=0)
    selected_sentence = pd.read_csv(reuslt_path2, sep='\t', header=None, index_col=0)
    selected_cid.rename(columns={1: 'cid'}, inplace=True)
    selected_sentence.rename(columns={1: 'sentence'}, inplace=True)
    result3 = pd.merge(selected_cid, selected_sentence, left_index=True, right_index=True)
    result3.to_csv(reuslt_path3, sep='\t', index_label='id')
