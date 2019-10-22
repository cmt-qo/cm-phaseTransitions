from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#tf.set_random_seed(5)
#tf.set_random_seed(6)
#np.random.seed(3)

L=8  #lattice size

# Hyper Parameters
batch_size = 50
LR = 0.00001 #0.00001         # learning rate
N_TEST_IMG = 300
index_in_epoch=0
num_examples=59950
epochs_completed=0
eval_size=50
training_=False

num_ex0=30

def extract_fn_apply_noise(data_record):
    sequence_features = {
        # Extract features using the keys set during creation
        'beta': tf.FixedLenFeature([], tf.float32),
        'loop': tf.FixedLenFeature([], tf.string)
        # If size is different of different records, use VarLenFeature 
        #'float_list2': tf.VarLenFeature(tf.float32)
    }
    sample = tf.parse_single_example(data_record, sequence_features)
    loop=tf.decode_raw(sample["loop"],tf.float64)
    #image=tf.decode_raw(sample["map"],tf.float32)
    #labels=tf.stack([sample["omegam"],sample["sigma8"]])
    return sample['beta'], loop

def next_batch(batch_size,index_in_epoch_,epochs_completed_,train_data_, train_labels_beta_):
    """Return the next `batch_size` examples from this data set."""
    start = index_in_epoch_
    index_in_epoch_ += batch_size
    if index_in_epoch_ > num_examples or (epochs_completed==0 and start==0):
      # Finished epoch
      if start!=0:
        epochs_completed_ += 1
      # Shuffle the data
      perm = np.arange(num_examples)
      np.random.shuffle(perm)
      train_data_ = train_data_[perm]
      #train_labels_ = train_labels_[perm]
      train_labels_beta_=train_labels_beta_[perm]
      # Start next epoch
      start = 0
      index_in_epoch_ = batch_size
    end = index_in_epoch_
    return train_data_[start:end], train_labels_beta_[start:end], index_in_epoch_,epochs_completed_,train_data_,train_labels_beta_

    
nbetas=100
ncategories=3





##################################################################################################
#network
###################################################################################################
# Mnist digits
# tf placeholder


tf_x = tf.placeholder(tf.float32, [None, L,L,2])    # value in the range of (0, 1)
#y = tf.placeholder(tf.int64, [None,1])
tf_y_beta=tf.placeholder(tf.float32, [None,1])


en00=tf.layers.conv2d(tf_x,100,kernel_size=(3,3))
en01=tf.layers.conv2d(en00,100,padding='same',kernel_size=(2,2))
en02=tf.reshape(en01,[-1,(L-2)*(L-2)*100])
en0 = tf.layers.dense(en02, 100, tf.nn.relu,name='en0_')  #128


en2=tf.layers.dropout(en0, rate=0.15, training=training_)

beta_output=tf.layers.dense(en2, 1)

#beta_output=tf.layers.dense(en23, 1)
tf_y_beta2=tf.cast(tf_y_beta,tf.float64)

loss=tf.losses.mean_squared_error(tf_y_beta2,beta_output)


#decoded=tf.nn.softmax(logits)
predictions=beta_output
#printpredictions=tf.Print(predictions, [predictions])

train = tf.train.AdamOptimizer(LR).minimize(loss)


#weights = tf.get_default_graph().get_tensor_by_name(os.path.split(en0.name)[0] + '/kernel:0')
    
saver=tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

##########################################################################################################

if training_:
   train_data=np.loadtxt("train_data/train_data.txt")
   train_data=np.reshape(train_data,(np.shape(train_data)[0],L,2,L))
   train_data=np.swapaxes(train_data,2,3)
   train_labels_beta=np.zeros((np.shape(train_data)[0],1))
   train_labels_beta[:,0]=np.loadtxt("train_data/train_labels_beta.txt")
   eval_data=np.loadtxt("train_data/eval_data.txt")
   eval_data=np.reshape(eval_data,(np.shape(eval_data)[0],L,2,L))
   eval_data=np.swapaxes(eval_data,2,3)
   eval_labels_beta=np.zeros((np.shape(eval_data)[0],1))
   eval_labels_beta[:,0]=np.loadtxt("train_data/eval_labels_beta.txt")
   eval_loss=[]
   train_loss=[]
   train_lossstates=[]
   train_lossbeta=[]
   eval_lossstates=[]
   eval_lossbeta=[]
   eval_perclossbeta=[]


   eval_accuracy_sb=[]
   eval_accuracy_beta_sb=[]
   eval_losssb=[]

   eval_lossstatessb=[]
   eval_lossbetasb=[]
   eval_perclossbetasb=[]
      
   
   #tf.reset_default_graph()
   #saver.restore(sess, 'slstates_dense_k3/-49900')
   for step in range(20000): #->evtl epochs hier schon drin, falls steps>size dataset 1200000
       #print np.shape(logits_)
       
       b_x, b_y_beta, index_in_epoch,epochs_completed,train_data,train_labels_beta= next_batch(batch_size,index_in_epoch,epochs_completed,train_data, train_labels_beta)
       _, loss_= sess.run([train, loss], feed_dict={tf_x:b_x, tf_y_beta: b_y_beta})
       
       if step % 100 == 0:     # plotting
           
           print('step: ',  step)
           print('train loss: %.4f', loss_)
           train_loss.append(loss_)

           beta_output_, loss2= sess.run([beta_output, loss], feed_dict={tf_x:eval_data, tf_y_beta:eval_labels_beta})
           #ybeta2_=np.reshape(ybeta2_,(eval_size,1))
           #accuracy =tf.metrics.accuracy(eval_labels,predictions_)
           eval_loss.append(loss2)
           

           print("eval loss: ", loss2) #, "perclossbeta: ", perclossbeta_
           saver.save(sess, 'checkpoints_new/', global_step=step)
           #print np.shape(decoded_data), np.shape(eval_labels)
           #print "eval accuracy: ", eval_accuracy_
              
   x=np.linspace(0,20000,np.shape(train_loss)[0])
   plt.figure()
   plt.plot(x,train_loss, label="train")
   plt.plot(x,eval_loss, label="eval")
   plt.xlabel('training step')
   plt.ylabel('loss')
   plt.legend(loc=1)
   plt.savefig("loss.png")
   
   
else:
    tf.reset_default_graph()
    saver.restore(sess, 'checkpoints/-19900')
    pbetas_mean=np.zeros(200)
    plabels_mean=np.zeros(200)
    pdiv_mean=np.zeros(200)
    plt.figure()
    
    num_predexamples=2000.0
    for p_i in range(int(num_predexamples)):
      pdata=np.loadtxt('pred_data/pdata{}.txt'.format(p_i))
      pdata=np.reshape(pdata,(np.shape(pdata)[0],L,2,L))
      pdata=np.swapaxes(pdata,2,3)
      plabels=np.loadtxt('pred_data/plabels{}.txt'.format(p_i))

      pbetas = sess.run([beta_output], feed_dict={tf_x: pdata})
      pbetas=np.reshape(pbetas,(np.shape(plabels)))   
      pdiv=np.zeros(np.shape(plabels))
      deltabeta=(plabels[1]-plabels[0])
      pbetas_mean+=pbetas
      plabels_mean+=plabels
      
      
    pbetas_mean/=num_predexamples
    plabels_mean/=num_predexamples
    pdelta=-plabels_mean+pbetas_mean

    #nint: avoiding too spiky behaviour by enlarging interval between two betas
    nint=6
    pdeltadiv=np.zeros(np.shape(plabels)[0]//nint)
    print(np.shape(pdeltadiv))
    plab=np.zeros_like(pdeltadiv)
    for i in range(np.shape(plabels)[0]//nint):
          if (i==0):
            pdeltadiv[i]=(pdelta[nint*i+nint]-pdelta[nint*i])/(nint*deltabeta) 
            plab[i]=plabels_mean[nint*i]
          else:
            pdeltadiv[i]=(pdelta[nint*i]-pdelta[nint*(i-1)])/(nint*deltabeta)
            plab[i]=plabels_mean[nint*i-nint//2]
            
    pdiv_mean/=num_predexamples
    #plt.plot(plabels_mean,pdelta)
    #plab=plab/1.2
    plt.plot(plab,pdeltadiv)
   
    b=np.argmax(pdeltadiv)
    print("Phase transition indicated by network is at beta=",plab[b]," (divergence of derivative).")
    #plt.plot(np.ones(20)*x[b],np.linspace(0,5,20))
    plt.xlabel("beta")
    plt.ylabel('derivative')
    plt.savefig("divergence.png")
    
