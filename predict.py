from PIL import Image
import tensorflow as tf
import tflib
import tflib.ops
import tflib.network
from tqdm import tqdm
import numpy as np
import data_loaders
import time
import os
import json
import sys
import pyperclip
sys.path.append('./im2markup/scripts/utils')
from image_utils import *
import subprocess
import glob

BATCH_SIZE      = 2
EMB_DIM         = 80
ENC_DIM         = 256
DEC_DIM         = ENC_DIM*2
NUM_FEATS_START = 64
D               = NUM_FEATS_START*8
V               = 502
NB_EPOCHS       = 50
H               = 20
W               = 50

X = tf.placeholder(shape=(None,None,None,None),dtype=tf.float32)
mask = tf.placeholder(shape=(None,None),dtype=tf.int32)
seqs = tf.placeholder(shape=(None,None),dtype=tf.int32)
learn_rate = tf.placeholder(tf.float32)

ctx = tflib.network.im2latex_cnn(X,NUM_FEATS_START,True)
out,state = tflib.ops.FreeRunIm2LatexAttention('AttLSTM',ctx,EMB_DIM,V,ENC_DIM,DEC_DIM,D,H,W)
predictions = tf.argmax(out[:,:,-V:],axis=2)

sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=8))
init = tf.global_variables_initializer()
sess.run(init)

weights = np.load('weights_best_numpy.npy').tolist()
var_dict = {x.name: x for x in tf.get_collection('variables')}
for key in var_dict.keys():
    if key not in weights.keys():
        print key," not found!!"
    else:
        sess.run(var_dict[key].assign(weights[key]))
        print "Initialized ",key
sess.run(var_dict['RNN/while/Embedding/Embedding:0'].assign(weights['Embedding/Embedding:0']))
sess.run(var_dict['RNN/while/MLP.1/MLP.1.W:0'].assign(weights['MLP.1/MLP.1.W:0']))
sess.run(var_dict['RNN/while/MLP.1/MLP.1.b:0'].assign(weights['MLP.1/MLP.1.b:0']))
properties = np.load('properties.npy').tolist()
def show():
    batch_size=1
    imgs = np.asarray(Image.open('tmp3.png').convert('YCbCr'))[:,:,0][None,None,:]
    inp_seqs = np.zeros((batch_size,160)).astype('int32')
    inp_seqs[:,0] = properties['char_to_idx']['#START']
    tflib.ops.ctx_vector = []

    idx_to_chars = lambda Y: ' '.join(map(lambda x: properties['idx_to_char'][x],Y))
    visualize=False
    inp_seqs = sess.run(predictions,feed_dict={X:imgs})

    str = idx_to_chars(inp_seqs.flatten().tolist()).split('#END')[0].replace('\left','').replace('\\right','').replace('&','')
    print "Latex sequence: ",str
    pyperclip.copy('$'+str+'$')

def run_demo(filename=None,scale=2):
    if filename:
        file = '/home/rithesh/Downloads/%s.pdf'%(filename)
    else:
        file = np.random.choice(glob.glob('Papers_PDF/*'))
    os.system('xdg-open ' + file)
    num = str(input("Enter page number: ")-1)
    os.system('convert -density 200 -quality 100 %s tmp.png'%(file+'[%s]'%num))
    Image.open('tmp.png').show()
    time.sleep(3)
    os.system('import screenshot.png')
    Image.open('screenshot.png').show()
    while raw_input("Is the crop correct? (y/n) : ").lower() not in ['y','yes']:
        os.system('pkill display')
        Image.open('tmp.png').show()
        time.sleep(3)
        os.system('import screenshot.png')
        os.system('pkill display')
        Image.open('screenshot.png').show()
    os.system('pkill display')

    status = crop_image('screenshot.png', './tmp1.png', (600,60))
    buckets = json.loads('[[240,100], [320,80], [400,80],[400,100], [480,80], [480,100], [560,80], [560,100], [640,80],[640,100], [720,80], [720,100], [720,120], [720, 200], [800,100],[800,320], [1000,200]]')
    buckets_2 = json.loads('[[120,50], [160,40], [200,40],[200,50], [240,40], [240,50], [280,40], [280,50], [320,40],[320,50], [360,40], [360,50], [360,60], [360, 100], [400,50],[400,160], [500,100]]')
    status = pad_group_image('./tmp1.png', './tmp2.png', [8,8,8,8], buckets)
    status = downsample_image('./tmp2.png', './tmp3.png', scale)
    show()

run_demo()
