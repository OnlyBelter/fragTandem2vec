# molFrag2vec
fragmentation and vectorization of small molecules


## 1. Getting started (tested on Ubuntu 18.04 )
### 1.1 clone github repo
```
git clone https://github.com/OnlyBelter/molFrag2vec.git
```

### 1.2 download Miniconda and install dependencies
- download miniconda, please see https://conda.io/en/master/miniconda.html
- also see: [Building identical conda environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#building-identical-conda-environments)
```
cd /where/is/molFrag2vec/located
conda create --name molFrag2vec --file requirements.txt
```

### 1.3 activate environment just created
```
conda activate molFrag2vec
# install rdkit and mol2vec
conda install -c rdkit rdkit==2019.03.3.0
pip install git+https://github.com/samoturk/mol2vec
```

### 1.4 building fastText for Python
- also see: https://github.com/facebookresearch/fastText#building-fasttext-for-python
```
$ git clone https://github.com/facebookresearch/fastText.git
$ cd fastText
$ git checkout tags/v0.9.1
$ pip install .
$ cd ..
$ rm -rf fastText
```

## 2. Molecule fragment and refragment
### 2.1 fragment
```
python mol_tree.py demo_data/demo_dataset.txt demo_data/step1_result.txt --log_fn demo_data/step1_log.log
```

### 2.2 refragment
```
python refragment.py ./demo_data/step1_result.txt ./demo_data/step2_result.txt --log_fn ./demo_data/step2_log.log
# plot molecular structure, molecular tree and molecular with index of the first 10 lines under test model
python refragment.py ./demo_data/step1_result.txt ./demo_data/step2_result.txt --log_fn ./demo_data/step2_log.log --test
```

### 2.3 get molecular vector, nearest neighbor and class

#### download following files from:

- s2_trained_model_molFrag2vec3.bin: https://doi.org/10.6084/m9.figshare.11589477
- s2_trained_model_kmeans_model.pkl: https://doi.org/10.6084/m9.figshare.11589477
- mol2vec.csv: https://doi.org/10.6084/m9.figshare.11589870
- pure_kmeans_class.csv: https://doi.org/10.6084/m9.figshare.11599773

and save in directory: molFrag2vec/big-data/...

```
# only get molecular vectors
python class_prediction.py ./big-data/s2_trained_model_molFrag2vec3.bin ./big-data/s2_trained_model_kmeans_model.pkl ./demo_data/step2_result.txt --result_dir ./demo_data --log_fn ./demo_data/step3_log.log

# calculate nearest neighbors
# need mol2vec.csv file, add by --training_mol_vec_fp parameter
python class_prediction.py ./big-data/s2_trained_model_molFrag2vec3.bin ./big-data/s2_trained_model_kmeans_model.pkl ./demo_data/step2_result.txt --result_dir ./demo_data --log_fn ./demo_data/step3_log.log --training_mol_vec_fp ./big-data/mol2vec.csv --find_nearest_neighbors

# predict class
python class_prediction.py ./big-data/s2_trained_model_molFrag2vec3.bin ./big-data/s2_trained_model_kmeans_model.pkl ./demo_data/step2_result.txt --result_dir ./demo_data --log_fn ./demo_data/step3_log.log --pure_kmeans_class_fp ./big-data/pure_kmeans_class.csv --predict_class
```
