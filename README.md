# A Pytorch Implementation of Tacotron: End-to-end Text-to-speech Deep-Learning Model
[INFO - 20190514] I will update this project in a few weeks. The performance of the new version code is much better than the current version. 

Implement google's [Tacotron](https://arxiv.org/abs/1703.10135) TTS system with pytorch. This is just my preliminary work, so there are many parts could be improved. I will boost the performance when I have time.  
![tacotron](arch_fig.jpg)

## Updates
2018/09/15: Fix RNN feeding bug.  
2018/11/04: Add attention mask and loss mask.


## Requirements
Download python and pytorch.  
* python==3.6.5
* pytorch==0.4.1  

You can use requirements.txt to download packages below.
```bash
# I recommend you use virtualenv.
$ pip install -r requirements.txt
```
* librosa  
* numpy  
* pandas  
* scipy  
* matplotlib  


## Usage

* Data  
Download [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) provided by keithito. It contains 13100 short audio clips of a single speaker. The total length is approximately 20 hrs.

* Set config.    
```python
# Set the 'meta_path' and 'wav_dir' in `hyperparams.py` to paths of your downloaded LJSpeech's meta file and wav directory.
meta_path = 'Data/LJSpeech-1.1/metadata.csv'
wav_dir = 'Data/LJSpeech-1.1/wavs'
```

* Train
```bash
# If you have pretrained model, add --ckpt <ckpt_path>
$ python main.py --train --cuda
```

* Evaluate 
```bash
# You can change the evaluation texts in `hyperparams.py`
# ckpt files are saved in 'tmp/ckpt/' in default
$ python main.py --eval --cuda --ckpt <ckpt_timestep.pth.tar>
```

## Samples
The sample texts is based on [Harvard Sentences](http://www.cs.columbia.edu/~hgs/audio/harvard.html). See the samples at `samples/` which are generated after training 200k.

## Alignment
The model starts learning something at 30k.
![alignment](alignment.gif)


## Differences from the original Tacotron
1. Data bucketing (Original Tacotron used loss mask)
2. Remove residual connection in decoder_CBHG
3. Batch size is set to 8
4. Gradient clipping
5. Noam style learning rate decay (The mechanism that [Attention is all you need](https://arxiv.org/abs/1706.03762) applies.)


## Refenrence
1. (Tensorflow) Kyubyong's  [implementation](https://github.com/Kyubyong/tacotron)
2. (Tensorflow) acetylSv's  [implementation](https://github.com/acetylSv/GST-tacotron)
3. (Pytorch)    soobinseo's [implementaition](https://github.com/soobinseo/Tacotron-pytorch)  

Finally, I have to say this work is highly based on Kyubyong's work, so if you are a tensorflow user, you may want to see his work. Also, feel free to give some feedbacks!
