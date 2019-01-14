# ML2018FALL FINAL REPORT

## Tool and Version
Here is my enviroment.
* CUDA10.0
* python 3.5.2
* pytorch 0.41
* pytorchvision 0.21
* pandas 0.23.4
* Augmentor


## Download dataset
Follow https://github.com/Kaggle/kaggle-api

and enter this command:
```
mkdir data
kaggle competitions download -c human-protein-atlas-image-classification
mkdir -p train test
unzip train.zip -d train
unzip test.zip -d test
cd ..
```

and your dictionary tree should look like this:
```
data/
├── sample_submission.csv
├── train.csv
├── train
│   └── xxx.png
└── test
    └── xxx.png
```

## Reproduce 
To reproduce kaggle score = 0.472 in private scoreboard.

Just enter:
`bash kaggle472.sh`

and `test30_22,test34_30,test35_36,test36_40,test37_33,test41_32,mean.csv` is the output.

## Usage
`python3 test.py [your .pt file]`

ex:

`python3 test.py test34_40`

and `test34_40.csv` will generated in the same folder.

## Ensemble
You can change any model you what to ensemble in the head of `ensemble.py` file.

and run
`python3 ensemble.py`

# Training
Change parameters in `train.py`, and make sure you have GPU.

and run 
`python3 run.py`

It will save model's weight automatically every epoches, and also save training history in `.npy' file.
