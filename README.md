# FuHLDR
## paper "Fusing Higher and Lower-order Biological Information for Drug Repositioning via Graph Representation Learning"

### 'data' directory
Contain B-Dataset and F-Dataset

### How to generate semantic embeddings?
The semantic embeddings, i.e. $\mathcal{Z}$ in the paper, are generated by metapath2vec algorithm. Users may refer to https://github.com/dmlc/dgl/tree/master/examples/pytorch/metapath2vec for an implementation.

### main.py
To obtain train and test data, run
  - python main.py -d 2
  - -d is dataset selection, which B-Dataset is represented as 1 and F-Dataset is represented as 2.

###
To predict drug-disease associations by FuHLDR, run
  - RVFL.m

### Options
See help for the other available options to use with *FuHLDR*
  - python FuHLDR.py --help

### Requirements
FuHLDR is tested to work under Python 3.6.2  
The required dependencies for FuHLDR are Keras, PyTorch, TensorFlow, numpy, pandas, scipy, and scikit-learn.

### Contacts
If you have any questions or comments, please feel free to email BoWei Zhao (stevejobwes@gmail.com).
