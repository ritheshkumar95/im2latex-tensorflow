import tensorflow as tf
import tflib
import tflib.ops
import tflib.network
from tqdm import tqdm
import numpy as np
import data_loaders
import time
import os

BATCH_SIZE      = 32
EMB_DIM         = 80
ENC_DIM         = 256
DEC_DIM         = ENC_DIM*2
NUM_FEATS_START = 64
D               = NUM_FEATS_START*8
V               = 502
NB_EPOCHS       = 50
H               = 20
W               = 50

# with tf.device("/cpu:0"):
#     custom_runner = data_loaders.CustomRunner()
#     X,seqs,mask,reset = custom_runner.get_inputs()
#
# print X,seqs
X = tf.placeholder(shape=(None,None,None,None),dtype=tf.float32)
mask = tf.placeholder(shape=(None,None),dtype=tf.int32)
seqs = tf.placeholder(shape=(None,None),dtype=tf.int32)
input_seqs = seqs[:,:-1]
target_seqs = seqs[:,1:]
emb_seqs = tflib.ops.Embedding('Embedding',V,EMB_DIM,input_seqs)

ctx = tflib.network.im2latex_cnn(X,NUM_FEATS_START,True)
out,state = tflib.ops.im2latexAttention('AttLSTM',emb_seqs,ctx,EMB_DIM,ENC_DIM,DEC_DIM,D,H,W)
logits = tflib.ops.Linear('MLP.1',out,DEC_DIM,V)


loss = tf.reshape(tf.nn.sparse_softmax_cross_entropy_with_logits(
    tf.reshape(logits,[-1,V]),
    tf.reshape(seqs[:,1:],[-1])
    ), [BATCH_SIZE, -1])

mask_mult = tf.to_float(mask[:,1:])
loss = tf.reduce_sum(loss*mask_mult)/tf.reduce_sum(mask_mult)

#train_step = tf.train.AdamOptimizer(1e-2).minimize(loss)

optimizer = tf.train.GradientDescentOptimizer(1e-1)
gvs = optimizer.compute_gradients(loss)
capped_gvs = [(tf.clip_by_norm(grad, 5.), var) for grad, var in gvs]
train_step = optimizer.apply_gradients(capped_gvs)

test_img = tf.placeholder(shape=(None,None,None,None),dtype=tf.float32)
test_seqs = tf.placeholder(shape=(None,None),dtype=tf.int32)
emb_seqs = tflib.ops.Embedding('Embedding',V,EMB_DIM,test_seqs)
ctx = tflib.network.im2latex_cnn(test_img,NUM_FEATS_START,True)
out,state = tflib.ops.im2latexAttention('AttLSTM',emb_seqs,ctx,EMB_DIM,ENC_DIM,DEC_DIM,D,H,W)
logits = tflib.ops.Linear('MLP.1',out,DEC_DIM,V)
predictions = tf.argmax(tf.nn.softmax(logits[:,-1]),axis=1)

def predict(batch_size=32):
    import random
    from PIL import Image
    # f = np.load('train_list_buckets.npy').tolist()
    f = np.load('test_buckets.npy').tolist()
    random_key = random.choice(f.keys())
    f = f[random_key]
    imgs = []
    while len(imgs)!=batch_size:
        start = np.random.randint(0,len(f),1)[0]
        # if os.path.exists('./images_processed/'+f[start][0]):
        #     imgs.append(np.asarray(Image.open('./images_processed/'+f[start][0]).convert('YCbCr'))[:,:,0][:,:,None])
        if os.path.exists('./images_processed/'+f[start]+'.png'):
            imgs.append(np.asarray(Image.open('./images_processed/'+f[start]+'.png').convert('YCbCr'))[:,:,0][:,:,None])

    imgs = np.asarray(imgs,dtype=np.float32).transpose(0,3,1,2)
    inp_seqs = np.zeros((batch_size,160)).astype('int32')
    inp_seqs[:,0] = np.load('properties.npy').tolist()['char_to_idx']['#START']
    for i in xrange(1,160):
        inp_seqs[:,i] = sess.run(predictions,feed_dict={test_img:imgs,test_seqs:inp_seqs[:,:i]})
        print i,inp_seqs[:,i]
    np.save('pred_imgs',imgs)
    np.save('pred_latex',inp_seqs)
    return inp_seqs

sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=8))
init = tf.global_variables_initializer()
#init = tf.initialize_all_variables()
sess.run(init)
saver = tf.train.Saver()
# start the tensorflow QueueRunner's
#tf.train.start_queue_runners(sess=sess)
# start our custom queue runner's threads
#custom_runner.start_threads(sess)

losses = []
times = []
print "Compiled Train and Test functions!"
#train_fn(np.random.randn(32,1,128,256).astype('float32'),np.random.randint(0,107,(32,50)).astype('int32'),np.random.randint(0,2,(32,50)).astype('int32'), np.zeros((32,1024)).astype('float32'))
i=0
for i in xrange(i,NB_EPOCHS):
    iter=0
    costs=[]
    times=[]
    itr = data_loaders.data_iterator(BATCH_SIZE)
    for train_img,train_seq,train_mask,train_reset in itr:
        iter += 1
        start = time.time()
        _ , _loss = sess.run([train_step,loss],feed_dict={X:train_img,seqs:train_seq,mask:train_mask})
        #_ , _loss = sess.run([train_step,loss])
        times.append(time.time()-start)
        costs.append(_loss)
        if iter%100==0:
            print "Iter: %d (Epoch %d)"%(iter,i+1)
            print "\tMean cost: ",np.mean(costs)
            print "\tMean time: ",np.mean(times)

    print "\n\nEpoch %d Completed!"%(i+1)
    print "\tMean cost: ",np.mean(costs)
    print "\tMean time: ",np.mean(times)
    print "\n\n"

#sess.run([train_step,loss],feed_dict={X:np.random.randn(32,1,256,512),seqs:np.random.randint(0,107,(32,40)),mask:np.random.randint(0,2,(32,40))})
