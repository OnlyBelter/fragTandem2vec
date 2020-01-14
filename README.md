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
conda create --name molFrag2vec --file spec-file.txt
```

### 1.3 activate environment just created
```
conda activate molFrag2vec
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
`
python class_prediction.py ./big-data/s2_trained_model_molFrag2vec3.bin ./big-data/s2_trained_model_kmeans_model.pkl ./demo_data/step2_result.txt --result_fp ./demo_data/step3_mol_vec.csv --log_fn ./demo_data/step3_log.log
`

