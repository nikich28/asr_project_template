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

## Load checkpoint
```shell
!FILEID='1NUwF5hQsGGQIaqKP1Z2fUIDhTFeWTJoY' && \
FILENAME='checkpoint-epoch42.pth' && \
FILEDEST="https://docs.google.com/uc?export=download&id=${FILEID}" && \
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate ${FILEDEST} -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILEID}" -O $FILENAME && rm -rf /tmp/cookies.txt
```


## Test my model

I use 42 epoch checkpoint for testing. So you can do it in DataSphere:

```shell
#!g1.1
%%python3 ./asr_project_template/test.py -c ./asr_project_template/hw_asr/dsmodel.json -r ./checkpoint-epoch42.pth
```

My best results on test data are:

WER: 0.49797065136972124

CER: 0.16205953704274267
