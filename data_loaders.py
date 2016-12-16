import tensorflow as tf
import time
import threading
import numpy as np
import re
import cv2
import glob
from PIL import Image

#properties = np.load('properties.npy').tolist()
#train_list = np.asarray(open("im2latex_train.lst","r").readlines())
#train_list = np.load('train_list.npy')
train_dict = np.load('train_list_buckets.npy').tolist()
print "Length of training data: ",len(train_dict)
# out_folder = glob.glob('./cropped_images/*')
# new_list = []
# for i in xrange(len(train_list)):
#     if './cropped_images/'+train_list[i].split()[1]+'.png' in out_folder:
#         new_list.append(train_list[i])
# print "Length of final training data: ",len(new_list)
# del train_list
# train_list = new_list

def old_data_iterator(batch_size = 32,SEQ_LEN=50):
    """ A simple data iterator """
    batch_idx = 0
    #while True:
    N_FILES = (len(train_list)//batch_size)*batch_size
    sorted_train_list = sorted(train_list,key=lambda x: len(x[1]),reverse=True)

    for batch_idx in range(0, N_FILES, batch_size):
        train_sublist = sorted_train_list[batch_idx:batch_idx+batch_size]
        imgs=[]
        batch_forms = []
        try:
            for data,forms in train_sublist:
                imgs.append(cv2.imread('./images_processed/'+data)[:,:,-1:])
                batch_forms.append(forms)
        except TypeError:
            continue
        imgs = np.asarray(imgs,dtype=np.float32).transpose(0,3,1,2)
        lens = [len(x) for x in batch_forms]
        mask = np.zeros((batch_size,max(lens)),dtype=np.int32)
        Y = np.zeros((batch_size,max(lens)),dtype=np.int32)
        for i,form in enumerate(batch_forms):
            mask[i,:len(form)] = 1
            Y[i,:len(form)] = form

        # for i in xrange(0,600,SEQ_LEN):
        #     if (mask[:,i]==0).all():
        #         break
        #     yield imgs, Y[:,i:i+SEQ_LEN], mask[:,i:i+SEQ_LEN], np.repeat(int(i==0),batch_size)
        yield imgs, Y, mask, np.repeat(1, batch_size)

    print "Epoch Completed!"

def data_iterator(batch_size = 32,SEQ_LEN=50):
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
            yield imgs, Y, mask, np.repeat(1, batch_size)

class CustomRunner(object):
    """
    This class manages the the background threads needed to fill
        a queue full of data.
    """
    def __init__(self,batch_size=32, SEQ_LEN=50):
        self.dataX = tf.placeholder(dtype=tf.float32, shape=[None, 1, 128, 256])
        self.dataY = tf.placeholder(dtype=tf.int32, shape=[None, None])
        self.dataMask = tf.placeholder(dtype=tf.int32, shape=[None, None])
        self.reset = tf.placeholder(tf.int32)

        # The actual queue of data. The queue contains a vector for
        # the mnist features, and a scalar label.
        self.queue = tf.RandomShuffleQueue(dtypes=[tf.float32, tf.int32, tf.int32, tf.int32],
                                           capacity=2000,
                                           min_after_dequeue=1000)

        self.SEQ_LEN = SEQ_LEN
        self.batch_size = batch_size

        # The symbolic operation to add data to the queue
        # we could do some preprocessing here or do it in numpy. In this example
        # we do the scaling in numpy
        self.enqueue_op = self.queue.enqueue_many([self.dataX, self.dataY, self.dataMask, self.reset])

    def get_inputs(self):
        """
        Return's tensors containing a batch of images and labels
        """
        images_batch, labels_batch, mask_batch, reset_batch = self.queue.dequeue_many(self.batch_size)
        return images_batch, labels_batch, mask_batch, reset_batch

    def thread_main(self, sess):
        """
        Function run on alternate thread. Basically, keep adding data to the queue.
        """
        for dataX, dataY, dataMask, reset in data_iterator(self.batch_size,self.SEQ_LEN):
            sess.run(self.enqueue_op, feed_dict={self.dataX:dataX, self.dataY:dataY, self.dataMask:dataMask, self.reset: reset})

    def start_threads(self, sess, n_threads=1):
        """ Start background threads to feed queue """
        threads = []
        for n in range(n_threads):
            t = threading.Thread(target=self.thread_main, args=(sess,))
            t.daemon = True # thread will close when parent quits
            t.start()
            threads.append(t)
        return threads
