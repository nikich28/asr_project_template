import json
import youtokentome as yttm

f = open('/home/jupyter/mnt/s3/bucket-hse-rw/data/datasets/librispeech/train-clean-100_index.json')
data = json.load(f)
with open("train-clean-100_index.txt", 'w') as fout:
    for d in data:
        t = d['text']
        print(t, file=fout)
f.close()


train_data_path = "train-clean-100_index.txt"
model_path = "bpe.model"

# Training model
yttm.BPE.train(data=train_data_path, vocab_size=200, model=model_path)

