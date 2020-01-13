# PubChem PUG REST
# http://pubchemdocs.ncbi.nlm.nih.gov/pug-rest
# https://pubchemdocs.ncbi.nlm.nih.gov/pug-rest
import io
import json
import numpy as np
import pandas as pd
import requests
import time


BASE_URL = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{}/property/{}/CSV'
BASE_URL2 = 'http://classyfire.wishartlab.com/entities/{}.json'


def get_property_by_cid(cid, mol_property, fn):
    """

    :param cid: a single cid or multiple cids sperated by
    :param mol_property: IsomericSMILES/ CanonicalSMILES/ MolecularFormula/ MolecularWeight/ InChIKey
    :return:
    """
    result = requests.get(BASE_URL.format(cid, mol_property))
    if result.status_code == 200:
        # https://stackoverflow.com/a/32400969/2803344
        content = result.content
        c = pd.read_csv(io.StringIO(content.decode('utf-8')))
        # print(c.to_dict())
        for i in range(c.shape[0]):
            with open(fn, 'a') as f:
                cid = c.loc[i, 'CID']
                prop_value = c.loc[i, mol_property]
                f.write('\t'.join([str(cid), prop_value]) + '\n')
    else:
        print('Getting the {} of cid {} is not successful.'.format(property, cid))
        with open('missing_pid.log', 'a') as f:
            f.write(cid + '\n')


def get_class_by_inchikey(cid, inchikey, fn):
    """
    get class by inchikey from classyFire
    :param cid:
    :param inchikey:
    :param fn:
    :return:
    """

    result = requests.get(BASE_URL2.format(inchikey))
    col_name = ['kingdom', 'superclass', 'class']
    values = [cid, inchikey]
    if result.status_code == 200:
        content = result.json()
        # print(type(content))
        # print(content)
        for col in col_name:
            # print(col)
            val = content.get(col, '')
            if val:
                val = val.get('name', '')
            values.append(val)
    elif result.status_code == 404:
        values.append('not found')
    elif result.status_code == 429:
        values.append('limit exceeded')
    with open(fn, 'a') as handle:
        handle.write('\t'.join([str(i) for i in values]) + '\n')


def sampling(fn, max_count, mass_range=(0, 1000)):
    """
    sampling cid from statistics result (organize cids by mass range)
    :param fn:
    :param max_count: max amount of cid needs to sample in each mass range
    :param mass_range: mass range
    :return:
    """
    mass2counter = {}
    with open(fn, 'r') as f:
        stat_result = json.load(f)
        sampling_result = {}
        for k, v in stat_result.items():
            if mass_range[0] <= int(k) <= mass_range[1]:
                if len(v) <= max_count:
                    sampling_result[k] = v
                    mass2counter[k] = len(v)
                else:
                    sampling_result[k] = list(np.random.choice(v, max_count, replace=False))
                    mass2counter[k] = max_count
                # print(k, len(sampling_result[k]))

    return {'sampling_result': sampling_result, 'class2counter': mass2counter}


def sampling_by_class_group(fn, max_count):
    """
    sampling cid from KMeans classification result (organize cids by class)
    :param fn:
    :param max_count: max amount of cid needs to sample in each mass range
    :return:
    """
    class2counter = {}
    mol2class = pd.read_csv(fn, index_col=0)
    mol2class_group = mol2class.groupby('mol_class')
    sampling_result = {}
    for k, v in mol2class_group:
        v = list(v.index.values)
        class2counter[k] = len(v)
        if len(v) <= max_count:
            sampling_result[k] = v
        else:
            sampling_result[k] = list(np.random.choice(v, max_count, replace=False))

    return {'sampling_result': sampling_result, 'class2counter': class2counter}


if __name__ == '__main__':
    # cid_file = 'big-data/mol2class_pca_kmeans_1000.csv'
    result_file = 'big-data/process/test_set_cid2InChIKey.txt'
    result_file2 = 'big-data/process/selected_test_set_cid2class_classyfire.txt'
    # result_file_new = 'big-data/cid2class_classyfire_supply_sup2.txt'
    log_file = 'big-data/download_status_class.log'
    sampling_test_set = 'big-data/process/test_set_sampling_12000.csv'
    # sampling_result = sampling_by_class_group(cid_file, max_count=100)
    # cids = sampling_result['sampling_result']
    # sampled_cids = pd.DataFrame.from_dict(cids, orient='index')
    # sampled_cids.to_csv('big-data/sampled_cids.csv', index_label='mol_class')
    # class2counter = pd.DataFrame.from_dict(sampling_result['class2counter'], orient='index')
    # class2counter.to_csv('big-data/kmean_class2counter.csv', index_label='mol_class')
    # selected_test_set = pd.read_csv(sampling_test_set)
    # cids = {0: list(selected_test_set['0'])}
    #
    # for k, v in cids.items():
    #     print('Downloading class {}, the number of cids: {}'.format(k, len(v)))
    #     cids_list = []
    #     if len(v) > 200:
    #         inx = np.arange(0, len(v), 200)
    #         for i in range(len(inx) - 1):
    #             cids_list.append(v[inx[i]:inx[i+1]])
    #             if (inx[i] == inx[-2]) and (inx[-1] < len(v)):
    #                 cids_list.append(v[inx[-1]:])
    #     else:
    #         cids_list.append(v)
    #     # print(cids_list)
    #     for _ in cids_list:
    #         _cids = ','.join([str(i) for i in _])
    #         # print(_cids)
    #         # time.sleep(1)
    #         get_property_by_cid(_cids, 'InChIKey', fn=result_file)
    #     with open(log_file, 'a') as log_f:
    #         log_f.write('class {} finished'.format(k) + '\n')

    cid2inchikey = pd.read_csv(result_file, header=None, sep='\t')
    # cid2inchikey = cid2inchikey[cid2inchikey[2] == 'limit exceeded'].copy()
    print(cid2inchikey.shape)
    counter = 0
    for i in cid2inchikey.index:
        # if counter % 5 == 0:
        #     time.sleep(1)
            # print(i, cid2inchikey.loc[i])
        if i >= 11144:
            # super_class = cid2inchikey.loc[i, 2]
            # print(super_class)
            # if super_class == 'nan':
            if counter % 20 == 0:
                print(i, cid2inchikey.loc[i])
            time.sleep(3)
            cid = cid2inchikey.loc[i, 0]
            inchikey = cid2inchikey.loc[i, 1]
            get_class_by_inchikey(cid, inchikey, result_file2)
        counter += 1
