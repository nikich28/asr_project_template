# ASR homework

## Installation guide

We do everything using DataSphere.

Firstly, install all necessary libs.

```shell
!git clone -b pipeline https://github.com/nikich28/asr_project_template

%pip install -r ./asr_project_template/requirements.txt
```

Also install ctcdecode for beam search via git:

```shell
!git clone --recursive https://github.com/parlance/ctcdecode.git

%cd ctcdecode
%pip install .

%cd /.
%cd home/jupyter/work/resources
```

## Training

For training you should use this command in Yandex DataSphere:
```shell
#!g1.1
%%python3 ./asr_project_template/train.py -c ./asr_project_template/hw_asr/dsmodel.json
```
You should train this model for at least 42 epochs.

## Test my model

I use 42 epoch checkpoint for testing. So you can do it in DataSphere:

```shell
#!g1.1
%%python3 ./asr_project_template/test.py -c ./asr_project_template/hw_asr/dsmodel.json -r saved/models/ds_pipeline/1025_213219377295/checkpoint-epoch42.pth
```

My best results on test data are:

WER: 0.49797065136972124

CER: 0.16205953704274267
