import tensorflow as tf
import time
import threading
import numpy as np
import re
import cv2
import glob
from PIL import Image

def data_iterator(set='train',batch_size = 32):
    '''
    Python data generator to facilitate mini-batch training
    Arguments:
        set - 'train','valid','test' sets
        batch_size - integer (Usually 32,64,128, etc.)
    '''
    train_dict = np.load(set+'_buckets.npy').tolist()
    print "Length of %s data: "%set,np.sum([len(train_dict[x]) for x in train_dict.keys()])

    for keys in train_dict.keys():
        train_list = train_dict[keys]
        N_FILES = (len(train_list)//batch_size)*batch_size
        for batch_idx in xrange(0,N_FILES,batch_size):
            train_sublist = train_list[batch_idx:batch_idx+batch_size]
            imgs = []
            batch_forms = []
            for x,y in train_sublist:
                imgs.append(np.asarray(Image.open('./images_processed/'+x).convert('YCbCr'))[:,:,0][:,:,None])
                batch_forms.append(y)
            imgs = np.asarray(imgs,dtype=np.float32).transpose(0,3,1,2)
            lens = [len(x) for x in batch_forms]

            mask = np.zeros((batch_size,max(lens)),dtype=np.int32)
            Y = np.zeros((batch_size,max(lens)),dtype=np.int32)
            for i,form in enumerate(batch_forms):
                mask[i,:len(form)] = 1
                Y[i,:len(form)] = form
            yield imgs, Y, mask

## Deprecated!! Queue Runners cannot be used as image is of variable size
class CustomRunner(object):
    """
    This class manages the the background threads needed to fill
        a queue full of data.
    """
    def __init__(self,batch_size=32, SEQ_LEN=50):
        self.dataX = tf.placeholder(dtype=tf.float32, shape=[None, 1, 128, 256])
        self.dataY = tf.placeholder(dtype=tf.int32, shape=[None, None])
        self.dataMask = tf.placeholder(dtype=tf.int32, shape=[None, None])

        # The actual queue of data. The queue contains a vector for
        # the mnist features, and a scalar label.
        self.queue = tf.RandomShuffleQueue(dtypes=[tf.float32, tf.int32, tf.int32],
                                           capacity=2000,
                                           min_after_dequeue=1000)

        self.SEQ_LEN = SEQ_LEN
        self.batch_size = batch_size

        # The symbolic operation to add data to the queue
        # we could do some preprocessing here or do it in numpy. In this example
        # we do the scaling in numpy
        self.enqueue_op = self.queue.enqueue_many([self.dataX, self.dataY, self.dataMask])

    def get_inputs(self):
        """
        Return's tensors containing a batch of images and labels
        """
        images_batch, labels_batch, mask_batch = self.queue.dequeue_many(self.batch_size)
        return images_batch, labels_batch, mask_batch

    def thread_main(self, sess):
        """
        Function run on alternate thread. Basically, keep adding data to the queue.
        """
        for dataX, dataY, dataMask in data_iterator(self.batch_size,self.SEQ_LEN):
            sess.run(self.enqueue_op, feed_dict={self.dataX:dataX, self.dataY:dataY, self.dataMask:dataMask})

    def start_threads(self, sess, n_threads=1):
        """ Start background threads to feed queue """
        threads = []
        for n in range(n_threads):
            t = threading.Thread(target=self.thread_main, args=(sess,))
            t.daemon = True # thread will close when parent quits
            t.start()
            threads.append(t)
        return threads
