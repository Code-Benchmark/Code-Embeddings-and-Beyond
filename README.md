## Description
This paper systemically evaluates the performance of nine program representation learning models on three common tasks, where seven models are based on abstract syntax trees and two models are based on plain text of source code. The tasks are code classification, code clone detection, and code search, respectively. The results of performance evaluation show that they perform diversely in each task and the performance of the AST-based models is generally unstable over different tasks. In order to further explain the results, we apply a prediction attribution technique to find what elements are captured by the models and responsible for the predictions in each task. 

## Environment

Python 3.6.8  
torch 1.7.1  
GPU: NVIDIA Tesla P40 GPU  
CUDA 10.2   

## Usage

1. Install the required dependencies `pip install -r requirements.txt`.

2. Process datasets according to different models.

3. To train the model in the directory of each task:

   ```bash
   python main.py --model_name tbcnn --batch_size 1 --epoch 50 --lr 0.01 --dataset_directory TBCNN/data --model_path data --USE_GPU True
   ```


## Structure

```
ICSME2022-main
├─ README.md
├─ requirements.txt
├─ attribution_sample.py
├─ code_classification
│	   ├─ main.py
|	   ├─ Dataloaders
|      |    ├─ dataloaders.py
|      |    └─ utils.py
|      ├─ code2vec
|      |	├─ data
|      |    ├─ model
|	   |	├─ data_process.py
|	   |	├─ train_and_test.py			
|	   |	└─ model.py
|      └─ ...
├─ code_clone_detection
├─ code_search
└─ model_implementation
       ├─ baselines
       ├─ TBCNN
       |	├─ ...
       ├─ TreeCaps
       |	├─ ...
       ├─ code2vec
       |    ├─ data_process.py
       |    └─ model.py
       ├─ code2seq
       |	├─ ...
       ├─ GGNN
       |  	├─ ...
       ├─ ASTNN
       |  	├─ ...
       └─ TPTrans
      		└─ ...

```


## Model Implementation
Our re-implementation for the evaluated models are .

Models with original source code in pytorch:

- ASTNN: https://github.com/zhangj111/astnn
- TPTrans: https://github.com/AwdHanPeng/TPTrans

Models with original source code in keras:

- code2vec: https://github.com/tech-srl/code2vec
- code2seq: https://github.com/tech-srl/code2seq

Models with original source code in Tensorflow:

- TBCNN: https://sites.google.com/site/treebasedcnn/home
- TreeCaps: https://github.com/bdqnghi/treecaps

Models without public available source (we refer some high stars implementation):

- GGNN: https://github.com/jacobwwh/graphmatch_clone; https://github.com/bdqnghi/ggnn.tensorflow


## Attribution Analysis
We upload a sample attribution code for ASTNN on Code Classification task.

