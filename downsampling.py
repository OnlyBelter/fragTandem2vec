import os
import numpy as np
import pandas as pd
from helper_func import get_class_md_combination


if __name__ == '__main__':
    root_dir = './big-data/moses_dataset/result/'
    file_name = 'mol2md_with_cid.csv'
    max_n = 5000
    result_fn = 'mol2md_downsampled_max_{}.csv'.format(max_n)
    cid2md = pd.read_csv(os.path.join(root_dir, file_name), usecols=list(range(1,10)), index_col='cid')
    print(cid2md.head(2))
    print('Start to get class of each molecule...')
    cid2class = get_class_md_combination(cid2md, min_number=1)
    print(cid2class.head(2))
    group_by_class = cid2class.groupby(['class'])
    class2count = group_by_class.count().loc[:, ['class_num']]
    # class2count.to_csv(os.path.join(root_dir, 'class2count.csv'))
    selected_cid = []
    for md_class in class2count.index:
        current_cid2class = cid2class[cid2class['class'] == md_class].copy()
        if class2count.loc[md_class, 'class_num'] <= max_n:
            pass
            # selected_cid.append(current_cid2class)
        else:
            current_cid2class = current_cid2class.sample(n=max_n, random_state=42)
        selected_cid.append(current_cid2class)
    selected_cid = pd.concat(selected_cid).index
    selected_cid2md = cid2md.loc[selected_cid,:].copy()
    print('The shape of selected cid2md: {}'.format(selected_cid2md.shape))
    selected_cid2md.to_csv(os.path.join(root_dir, result_fn))
