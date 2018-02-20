import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class TextClassifier():

    def __init__(self,  
                embed_dim,
                hid_dims,
                num_classes,
                 nonlin=tf.tanh,
                 optimizer=tf.train.AdamOptimizer(5e-4)):
        self.embed_dim = embed_dim
        self.hid_dims = hid_dims
        self.num_classes = num_classes
        self.nonlin = nonlin
        self.optimizer = optimizer
        
    def build_graph(self):
        if 'sess' in globals() and sess:
            sess.close()
        tf.reset_default_graph()
        is_train = tf.placeholder(tf.bool)
        x = tf.placeholder(tf.float32, [None,self.embed_dim])
        y = tf.placeholder(tf.int32,[None,self.num_classes])
        
        dims = [self.embed_dim]+self.hid_dims+[self.num_classes]
        
        for i,dim in enumerate(dims[1:]):
        
            W = tf.get_variable("W%i"%i, shape=[dims[i-1],dim])
            b = tf.get_variable("b%i"%i, shape=[dim])
            h = self.nonlin(tf.nn.xw_plus_b(x,W,b))

        U = tf.get_variable('U',shape=[self.hidden,self.num_classes])
        c = tf.get_variable('c',shape=[self.num_classes])
        preds = tf.nn.xw_plus_b(h,U,c)


        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(preds, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        loss = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=preds))

        train_step = self.optimizer.minimize(loss)
        
        self.x = x
        self.y = y
        self.loss = loss
        self.preds = preds
        self.train_step = train_step
        self.is_train = is_train
        self.accuracy = accuracy
        
        
    def train_graph(self,
                    data,
                    verbose=False,
                    batch_size = 50,
                    num_epochs = 100, 
                    skip_step=10):

        (x_train,y_train),(x_test,y_test) = [data['train'],data['test']]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            self.tr_accs, self.te_accs = [], []
            self.tr_losses, self.te_losses = [], []

            train_size = x_train.shape[0]

            for epoch in range(num_epochs):
                shuffle = np.random.permutation(np.arange(train_size))
                x_train = x_train[shuffle,:]
                y_train = y_train[shuffle,:]

                accuracy = 0
                losses = []

                for i in range(math.ceil(train_size/batch_size)):
                    ix = slice(i*50,(i+1)*50)
                    feed = {self.is_train: True,
                        self.x: x_train[ix,:],
                        self.y: y_train[ix,:]}

                    acc,_,loss = sess.run([self.accuracy,
                                          self.train_step,
                                          self.loss], 
                                          feed_dict=feed)

                    actual_batch_size = x_train[ix,:].shape[0]
                    losses.append(loss*actual_batch_size)
                    accuracy += acc*actual_batch_size

                feed = {self.is_train: True,
                        self.x: x_train[ix,:],
                        self.y: y_train[ix,:]}

                preds,val_acc,val_loss = sess.run([self.preds,
                                          self.train_step,
                                          self.loss], feed_dict=feed)

                total_tr_acc = accuracy/x_train.shape[0]
                total_tr_loss = np.sum(losses)/x_train.shape[0]
                if(verbose):
                    print("Epoch {2}, Overall training loss = {0:.3g} and accuracy of {1:.3g}"\
                      .format(total_tr_loss,total_tr_acc,epoch+1))
                    print("Epoch {2}, Overall validation loss = {0:.3g} and accuracy of {1:.3g}"\
                      .format(val_loss,val_acc,epoch+1))

                for val,lst in zip((total_tr_loss,val_loss,total_tr_acc,val_acc),
                                   (self.tr_losses,self.te_losses,self.tr_accs,self,self.te_accs)):
                    lst.append(val)
                    
        return self.tr_losses, self.te_losses, self.tr_accs, self.te_accs

    def plot(self):
            for i,j in zip(('Train Loss','Val Loss','Train Acc','Val Acc'),
                        [ self.tr_losses, self.te_losses, self.tr_accs, self.te_accs]):
                print(i,j[-1])
            plt.subplot(1,2,1)
            plt.plot(self.tr_losses,label='train')
            plt.plot(self.te_losses,label='valid')
            plt.title('Loss per epoch')
            plt.legend()
            plt.subplot(1,2,1)
            plt.plot(self.tr_accs,label='train')
            plt.plot(self.te_accs,label='valid')
            plt.title('Accuracy per epoch')
            plt.legend()

    def train_and_plot(self,
                    data,
                    verbose=False,
                    batch_size = 50,
                    num_epochs = 100, 
                    skip_step=10):
        self.train_graph()
        self.plot()
