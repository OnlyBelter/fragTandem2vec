# molFrag2vec
fragmentation and vectorization of small molecules


## 1. Getting started (tested on Ubuntu 18.04 )
### 1.1 clone github repo
```shell script
git clone https://github.com/OnlyBelter/molFrag2vec.git
```

### 1.2 download Miniconda and install dependencies
- download miniconda, please see https://conda.io/en/master/miniconda.html
- also see: [Building identical conda environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#building-identical-conda-environments)
```shell script
cd /where/is/molFrag2vec/located
conda create --name molFrag2vec --file requirements.txt
```

### 1.3 activate environment just created
```shell script
conda activate molFrag2vec
# install rdkit and mol2vec
conda install -c rdkit rdkit==2019.03.3.0
pip install git+https://github.com/samoturk/mol2vec
```

### 1.4 building fastText for Python
- also see: https://github.com/facebookresearch/fastText#building-fasttext-for-python
```shell script
$ git clone https://github.com/facebookresearch/fastText.git
$ cd fastText
$ git checkout tags/v0.9.1
$ pip install .
$ cd ..
$ rm -rf fastText
```

## 2. Molecule fragment and refragment
### 2.1 fragment
- The first step: generate molecular tree based on https://github.com/wengong-jin/icml18-jtnn.
  Then we can get the fragments of each molecule and the relation between each two fragments.
```shell script
python mol_tree.py demo_data/demo_dataset.txt demo_data/step1_result.txt --log_fn demo_data/step1_log.log
# or calculate molecular tree of examples in paper
python mol_tree.py dataset/examples_in_paper/cid2smiles.txt dataset/examples_in_paper/step1_result.txt --log_fn dataset/examples_in_paper/step1_log.log
```

### 2.2 refragment
a. generate fragment sentence

b. count fragment

c. replace fragment SMILES by fragment id
```shell script
python refragment.py ./demo_data/step1_result.txt ./demo_data/ --log_fn ./demo_data/step2_log.log
# plot molecular structure, molecular tree and molecular with index of the first 10 lines under test model
python refragment.py ./demo_data/step1_result.txt ./demo_data/ --log_fn ./demo_data/step2_log.log --test
```
#### output
- step2_cid2frag_id_sentence.csv: cid and fragment id sentence
- step2_cid2smiles_sentence.csv: cid and SMILES sentence
- step2_frag2num.csv: count the number of fragments in all dataset
- step2_frag_id_sentence.csv: only fragment id sentence separated by space, can be used to training model

### 2.3 training model
Training molFrag2vec model by FastText, and get the vectors of all fragments.
```shell script
python training_model.py ./demo_data/step2_frag_id_sentence.csv ./demo_data/step3_molFrag2vec_demo.bin
```
#### output
- step3_frag2vec_model_molFrag2vec.csv: fragment vectors
- step3_molFrag2vec_demo.bin: well-trained unsupervised frag2vec model

### 2.4 calculating molecular vector
Calculate molecular vector of selected molecules which have been classed by classyFire
```shell script
python smiles2vec.py
```

### 2.5 clustering
Cluster selected molecules by DBSCAN algorithm, and calculate purity score based on classyFire superclass
```shell script
python clustering.py './demo_data' './demo_data' --include_small_dataset_dir './dataset' --model molFrag2vec
```


### 2.6 get nearest neighbor and class

#### download following files from:

- s2_trained_model_molFrag2vec3.bin: https://doi.org/10.6084/m9.figshare.11589477
- s2_trained_model_kmeans_model.pkl: https://doi.org/10.6084/m9.figshare.11589477
- mol2vec.csv: https://doi.org/10.6084/m9.figshare.11589870
- pure_kmeans_class.csv: https://doi.org/10.6084/m9.figshare.11599773

and save in directory: molFrag2vec/big-data/...

```shell script
# only get molecular vectors
python class_prediction.py big-data/all_cid2smiles/x_training_set_cid2_sentence_new.csv big-data/frag2vec_model_fragTandem2vec_new.csv --result_dir big-data/all_cid2smiles/ --log_fn big-data/all_cid2smiles/class_prediction_log.log

# calculate nearest neighbors
# need mol2vec.csv file, add by --training_mol_vec_fp parameter
python class_prediction.py dataset/examples_in_paper/step2_cid2frag_id_sentence.csv big-data/frag2vec_model_fragTandem2vec_new.csv --result_dir dataset/examples_in_paper/ --training_mol_vec_fp big-data/all_cid2smiles/mol_vec_all_training_set_model_molFrag2vec.csv --log_fn dataset/examples_in_paper/class_prediction.log --find_nearest_neighbors

# predict class
# python class_prediction.py ./big-data/s2_trained_model_molFrag2vec3.bin ./demo_data/step2_result.txt --clustering_model_fp ./big-data/s2_trained_model_kmeans_model.pkl --result_dir ./demo_data --log_fn ./demo_data/step3_log.log --pure_kmeans_class_fp ./big-data/pure_kmeans_class.csv --predict_class
```

### 附件
附件1：所有碎片以及出现的次数，dataset/fragTandem2vec_related/frag2num_training_set_model_fragTandem2vec.csv
附件2：训练好的碎片向量模型
- Mol2vec model: dataset/mol2vec_related/mol2vec_model.pkl
- fragTandem2vec model: 