# A Pytorch Implementation of Tacotron: End-to-end Text-to-speech Deep-Learning Model
Implement google's [Tacotron](https://arxiv.org/abs/1703.10135) TTS system with pytorch. 
![tacotron](arch_fig.jpg)

## Updates
2018.09.15 : Fix RNN feeding bug.  
2018.11.04 : Add attention mask and loss mask.  
2019.05.17 : 2nd version updated.  


## Requirements
See `used_packages.txt`.


## Usage

* Data  
Download [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) provided by keithito. It contains 13100 short audio clips of a single speaker. The total length is approximately 24 hrs.

* Preprocessing
```bash
# Generate a directory 'training/' containing extracted features and a new meta file 'ljspeech_meta.txt'
$ python data/preprocess.py --output-dir training \ 
                            --data-dir <WHERE YOU PUT YOUR DATASET>/LJSpeech-1.1/wavs \
                            --old-meta <WHERE YOU PUT YOUR DATASET>/LJSpeech-1.1/metadata.csv \
                            --config config/config.yaml
```

* Split dataset
```bash
# Generate 'meta_train.txt' and 'meta_test.txt' in 'training/'
$ python data/train_test_split.py --meta-all training/ljspeech_meta.txt \ 
                                  --ratio-test 0.1
```

* Train
```bash
# Start training
$ python main.py --config config/config.yaml \
                 --checkpoint-dir <WHERE YOU WANT TO PUT YOUR CHECKPOINTS> 

# Restart training
$ python main.py --config config/config.yaml \
                 --checkpoint-dir <WHERE YOU WANT TO PUT YOUR CHECKPOINTS> \
                 --checkpoint-path <LAST CHECKPOINT PATH>
```

* Inference
```bash
# Generate synthesized speech 
$ python generate_speech.py --text <WHATEVER YOU WANT> \
                            --output <WHERE TO PUT OUTPUT FILE> \ 
                            --checkpoint-path <CHECKPOINT PATH> \
                            --config config/config.yaml
```


## Samples
I will update the samples later.

## Alignment
Proper alignment occurs after 10k steps of updating.


## Differences from the original Tacotron
1. Gradient clipping
2. Noam style learning rate decay (The mechanism that [Attention is all you need](https://arxiv.org/abs/1706.03762) applies.)


## Refenrence
[Original paper](https://arxiv.org/abs/1703.10135)  
[r9y9's implementation](https://github.com/r9y9/tacotron_pytorch)

Finally, this code is used in my work ["End-to-end Text-to-speech for Low-resource Languages by Cross-Lingual Transfer Learning"](https://arxiv.org/abs/1904.06508). 
