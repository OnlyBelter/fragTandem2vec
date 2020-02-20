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


def get_property_by_cid(cid, property, fn):
    """

    :param cid: a single cid or multiple cids sperated by ','
    :param property: IsomericSMILES/ MolecularFormula/ MolecularWeight/ InChIKey
    :return:
    """
    result = requests.get(BASE_URL.format(cid, property))
    if result.status_code == 200:
        # https://stackoverflow.com/a/32400969/2803344
        content = result.content
        c = pd.read_csv(io.StringIO(content.decode('utf-8')))
        # print(c.to_dict())
        for i in range(c.shape[0]):
            with open(fn, 'a') as f:
                cid = c.loc[i, 'CID']
                prop = c.loc[i, property]
                f.write('\t'.join([str(cid), prop]) + '\n')
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
    cid_file = 'big-data/cid2class_classyfire/mol2vec/unique_in_mol2vec.csv'
    result_file = 'big-data/cid2class_classyfire/mol2vec/cid2InChIKey.txt'
    result_file2 = 'big-data/cid2class_classyfire/mol2vec/cid2class_classyfire.txt'
    # result_file_new = 'big-data/cid2class_classyfire_supply4.txt'
    # log_file = 'big-data/download_status_class.log'

    # # fetch InChIKey by cid
    # cid_list = pd.read_csv(cid_file, index_col=0).index.tolist()
    # num_cid = len(cid_list)
    # for i in range(500):
    #     start_inx = i * 200
    #     end_inx = (i + 1) * 200
    #     if end_inx > 100:
    #         if end_inx <= num_cid:
    #             current_cids = ','.join([str(i) for i in cid_list[start_inx:end_inx]])
    #         else:
    #             current_cids = ','.join([str(i) for i in cid_list[start_inx:]])
    #         get_property_by_cid(cid=current_cids, property='InChIKey', fn=result_file)

    # fetch class by InChIKey from classyFire
    cid2inchikey = pd.read_csv(result_file, header=None, sep='\t')
    # cid2inchikey = cid2inchikey[cid2inchikey[2].isnull()].copy()
    print(cid2inchikey.shape)
    counter = 0
    for i in cid2inchikey.index:
        # if counter % 5 == 0:
        #     time.sleep(1)
            # print(i, cid2inchikey.loc[i])
        if i > 3406:
        # super_class = cid2inchikey.loc[i, 2]
        # print(super_class)
        # if super_class == 'nan':
            if counter % 50 == 0:
                print(i, cid2inchikey.loc[i])
            time.sleep(3)
            cid = cid2inchikey.loc[i, 0]
            inchikey = cid2inchikey.loc[i, 1]
            get_class_by_inchikey(cid, inchikey, result_file2)
            counter += 1
