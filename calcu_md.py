import os
import numpy as np
import pandas as pd
from helper_func import cal_md_by_smiles


if __name__ == '__main__':
    root_dir = 'big-data/moses_dataset/result'
    smiles_fn = 'step1_result.txt'
    # cid2smiles = pd.read_csv(os.path.join(root_dir, smiles_fn),
    #                          usecols=['cid', 'smiles'], index_col=0, sep='\t')
    # # cid2smiles = cid2smiles.to_string()
    #
    # print('There are {} lines (, cols) were read...'.format(cid2smiles.shape))
    # inx_range = np.arange(0, cid2smiles.shape[0], 100000)
    # inx_range = np.append(inx_range, cid2smiles.shape[0])
    # for inx in range(len(inx_range) - 1):
    #     print('Start to calculate {} part...'.format(inx))
    #     i0 = inx_range[inx]  # start index
    #     i1 = inx_range[inx + 1]  # end index
    #     current_cid2smiles = cid2smiles.iloc[i0:i1, :].copy()
    #     # print(current_cid2smiles.dtypes)
    #     # print(current_cid2smiles.head(3))
    #     smiles_list = current_cid2smiles.smiles.to_list()
    #     smiles2md = cal_md_by_smiles(smiles_list=smiles_list)
    #     # print(smiles2md.head(2))
    #     # print(smiles2md.dtypes)
    #     # cid2smiles = pd.concat([current_cid2smiles, smiles2md], axis=1)
    #     # cid2smiles = current_cid2smiles.merge(smiles2md, left_on='smiles', right_index=True)
    #     # cid2smiles.set_index('cid')
    #     if not os.path.exists(os.path.join(root_dir, 'mol2md.csv')):
    #         smiles2md.to_csv(os.path.join(root_dir, 'mol2md.csv'), mode='a', index_label='smiles')  # first write
    #     else:
    #         smiles2md.to_csv(os.path.join(root_dir, 'mol2md.csv'), mode='a', header=None)

    # add cid
    smiles2cid = pd.read_csv(os.path.join(root_dir, smiles_fn),
                             usecols=['cid', 'smiles'], index_col=1, sep='\t')
    # cid2smiles = cid2smiles.to_string()
    print('There are {} lines (, cols) were read...'.format(smiles2cid.shape))
    smiles2cid = smiles2cid['cid'].to_dict()
    smiles2md = pd.read_csv(os.path.join(root_dir, 'mol2md.csv'))
    smiles2md['cid'] = smiles2md['smiles'].apply(lambda x: smiles2cid[x])
    print(smiles2md.head(2))
    smiles2md.to_csv(os.path.join(root_dir, 'mol2md_with_cid.csv'), index=False)