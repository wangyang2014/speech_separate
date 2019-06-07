import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.ops import math_ops
import numpy as np
from confing import TIMESTEPS

learning_rata = 0.001
training_iters = 200000
batch_size = 10
display_step = 100
timesteps = TIMESTEPS

n_input = 129 
n_ouput = 120
num_hidden = 120 # hidden layer num of features
n_classes = 4
dropout = 0.75
n_lstminput = n_input + 120

indata = tf.placeholder(tf.float32,[None,n_input,TIMESTEPS])
otdata = tf.placeholder(tf.float32,[None,n_ouput,TIMESTEPS])
label = tf.placeholder(tf.float32,[None,n_classes,TIMESTEPS])
#lstmin = tf.placeholder(tf.float32,[None,n_lstminput])

_state = (tf.Variable(tf.random_normal([num_hidden])),tf.Variable(tf.random_normal([num_hidden])))

weight={'wc1': tf.Variable(tf.random_normal([num_hidden,n_lstminput])),
    'wc2': tf.Variable(tf.random_normal([num_hidden,n_lstminput])),
    'wc3': tf.Variable(tf.random_normal([num_hidden,n_lstminput])),
    'wc4': tf.Variable(tf.random_normal([num_hidden,n_lstminput])),
    'wi1': tf.Variable(tf.random_normal([num_hidden,n_lstminput])),
    'wi2': tf.Variable(tf.random_normal([num_hidden,n_lstminput])),
    'wi3': tf.Variable(tf.random_normal([num_hidden,n_lstminput])),
    'wi4': tf.Variable(tf.random_normal([num_hidden,n_lstminput])),
    'wf1': tf.Variable(tf.random_normal([num_hidden,n_lstminput])),
    'wf2': tf.Variable(tf.random_normal([num_hidden,n_lstminput])),
    'wf3': tf.Variable(tf.random_normal([num_hidden,n_lstminput])),
    'wf4': tf.Variable(tf.random_normal([num_hidden,n_lstminput])),
    'wo1': tf.Variable(tf.random_normal([num_hidden,n_lstminput])),
    'wo2': tf.Variable(tf.random_normal([num_hidden,n_lstminput])),
    'wo3': tf.Variable(tf.random_normal([num_hidden,n_lstminput])),
    'wo4': tf.Variable(tf.random_normal([num_hidden,n_lstminput])),
    'uc1': tf.Variable(tf.random_normal([num_hidden,num_hidden])),
    'uc2': tf.Variable(tf.random_normal([num_hidden,num_hidden])),
    'uc3': tf.Variable(tf.random_normal([num_hidden,num_hidden])),
    'uc4': tf.Variable(tf.random_normal([num_hidden,num_hidden])),
    'ui1': tf.Variable(tf.random_normal([num_hidden,num_hidden])),
    'ui2': tf.Variable(tf.random_normal([num_hidden,num_hidden])),
    'ui3': tf.Variable(tf.random_normal([num_hidden,num_hidden])),
    'ui4': tf.Variable(tf.random_normal([num_hidden,num_hidden])),
    'uf1': tf.Variable(tf.random_normal([num_hidden,num_hidden])),
    'uf2': tf.Variable(tf.random_normal([num_hidden,num_hidden])),
    'uf3': tf.Variable(tf.random_normal([num_hidden,num_hidden])),
    'uf4': tf.Variable(tf.random_normal([num_hidden,num_hidden])),
    'uo1': tf.Variable(tf.random_normal([num_hidden,num_hidden])),
    'uo2': tf.Variable(tf.random_normal([num_hidden,num_hidden])),
    'uo3': tf.Variable(tf.random_normal([num_hidden,num_hidden])),
    'uo4': tf.Variable(tf.random_normal([num_hidden,num_hidden]))
}
biases={
    'bwc1': tf.Variable(tf.random_normal([num_hidden,1])),
    'bwc2': tf.Variable(tf.random_normal([num_hidden,1])),
    'bwc3': tf.Variable(tf.random_normal([num_hidden,1])),
    'bwc4': tf.Variable(tf.random_normal([num_hidden,1])),
    'bwi1': tf.Variable(tf.random_normal([num_hidden,1])),
    'bwi2': tf.Variable(tf.random_normal([num_hidden,1])),
    'bwi3': tf.Variable(tf.random_normal([num_hidden,1])),
    'bwi4': tf.Variable(tf.random_normal([num_hidden,1])),
    'bwf1': tf.Variable(tf.random_normal([num_hidden,1])),
    'bwf2': tf.Variable(tf.random_normal([num_hidden,1])),
    'bwf3': tf.Variable(tf.random_normal([num_hidden,1])),
    'bwf4': tf.Variable(tf.random_normal([num_hidden,1])),
    'bwo1': tf.Variable(tf.random_normal([num_hidden,1])),
    'bwo2': tf.Variable(tf.random_normal([num_hidden,1])),
    'bwo3': tf.Variable(tf.random_normal([num_hidden,1])),
    'bwo4': tf.Variable(tf.random_normal([num_hidden,1]))
}

initValue={'ct':tf.Variable(tf.random_normal([num_hidden,1])),
    'ht':tf.Variable(tf.random_normal([num_hidden,1]))
}

def getAdd(mydist,str1,state):
    value = tf.zeros_like(mydist[str1+str(1)])
    for i in range(0,4):
        key = str1 + str(i+1)
        value = value + tf.multiply(mydist[key],state[0][i])
    
    return value
    


def LSTMCell(state,lastH,lastC,inputdata):
    sigmoid = math_ops.sigmoid
    tanh =  math_ops.tanh
    #logistic = math_ops.log_sigmoid()
    ''''Wc = weight['wc1'] * state[0] + weight['wc2'] * state[1] + weight['wc3'] * state[2] + weight['wc4'] * state[3]
    Wi = weight['wi1'] * state[0] + weight['wi2'] * state[1] + weight['wi3'] * state[2] + weight['wi4'] * state[3]
    Wf = weight['wf1'] * state[0] + weight['wf2'] * state[1] + weight['wf3'] * state[2] + weight['wf4'] * state[3]
    Wo = weight['wo1'] * state[0] + weight['wo2'] * state[1] + weight['wo3'] * state[2] + weight['wo4'] * state[3]
    Uc = weight['uc1'] * state[0] + weight['uc2'] * state[1] + weight['uc3'] * state[2] + weight['uc4'] * state[3]
    Ui = weight['ui1'] * state[0] + weight['ui2'] * state[1] + weight['ui3'] * state[2] + weight['ui4'] * state[3]
    Uf = weight['uf1'] * state[0] + weight['uf2'] * state[1] + weight['uf3'] * state[2] + weight['uf4'] * state[3]
    Uo = weight['uo1'] * state[0] + weight['uo2'] * state[1] + weight['uo3'] * state[2] + weight['uo4'] * state[3]

    bwc = biases['bwc1'] * state[0] + biases['bwc2'] * state[1] + biases['bwc3'] * state[2] + biases['bwc4'] * state[2]
    bwi = biases['bwi1'] * state[0] + biases['bwi2'] * state[1] + biases['bwi3'] * state[2] + biases['bwi4'] * state[2]
    bwf = biases['bwf1'] * state[0] + biases['bwf2'] * state[1] + biases['bwf3'] * state[2] + biases['bwf4'] * state[2]
    bwo = biases['bwo1'] * state[0] + biases['bwo2'] * state[1] + biases['bwo3'] * state[2] + biases['bwo4'] * state[2]'''

    Wc = getAdd(weight,'wc',state)
    Wf = getAdd(weight,'wf',state)
    Wi = getAdd(weight,'wi',state)
    Wo = getAdd(weight,'wo',state)
    Uc = getAdd(weight,'uc',state)
    Ui = getAdd(weight,'ui',state)
    Uf = getAdd(weight,'uf',state)
    Uo = getAdd(weight,'uo',state)
    bwc = getAdd(biases,'bwc',state)
    bwi = getAdd(biases,'bwi',state)
    bwf = getAdd(biases,'bwf',state)
    bwo = getAdd(biases,'bwo',state)
    inputdata = tf.reshape(inputdata,[n_lstminput,1])
    lastH = tf.reshape(lastH,[num_hidden,1])
    mct = tf.matmul(Wc,inputdata) + tf.matmul(Uc,lastH) + bwc
    mct = tanh(mct)

    it = tf.matmul(Wi,inputdata) + tf.matmul(Ui,lastH) + bwi
    it = sigmoid(it)

    ft = tf.matmul(Wf,inputdata) + tf.matmul(Uf,lastH) + bwf
    ft = sigmoid(ft)

    ot = tf.matmul(Wo,inputdata) + \
        tf.matmul(Uo,lastH) + bwo
    ot = sigmoid(ot)

    ct = tf.multiply(ft,lastC) +  tf.multiply(it,mct) 
    ht = tf.multiply(ot,tanh(ct))

    return ht,ct

def dynamic_rnn(inputs,state,lastH=None,lastC=None):
    outputslist = []
    for i in range(0,timesteps):
        if lastC is None:
            lastH,lastC = initValue['ht'],initValue['ct']
        ht,ct = LSTMCell(state[i],lastH,lastC,inputs[:,i])
        lastH,lastC = ht,ct
        outputslist.append(ht)
    return outputslist,ct

num_classes = 4
HMMweights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
HMMbiases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}
# Define a lstm cell with tensorflow
lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.reshape(x,[1,timesteps,n_input]) 
    x = tf.unstack(x, timesteps, 1)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    
    #rnn._Linear(outputs,batch_size,HMMbiases['out'])

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], HMMweights['out']) + HMMbiases['out'],outputs

#logits,outputs = RNN(x, HMMweights, HMMbiases)

StateLabel = tf.constant([[[1,0,0,0]],[[0,1,0,0]],[[0,0,1,0]],[[0,0,0,1]]])

def lossFunction(inputdata,y,outputdata):
    logits,outputs = RNN(inputdata, HMMweights, HMMbiases)
    state = []
    loss_op = 0.0
    for i in range(0,timesteps):
        logits = tf.matmul(outputs[i], HMMweights['out']) + HMMbiases['out']
        loss_op += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=y[:,i]))
        prediction = tf.nn.softmax(logits)
        state.append(prediction)
    out = tf.transpose(tf.concat(outputs,0))
    lstmin  = tf.concat([inputdata,out],0)
    outputslist,ct = dynamic_rnn(lstmin,state)
    #outputsArray = np.asarray(outputslist)
    outputsTF =  tf.convert_to_tensor(outputslist)
    outputsTF = tf.transpose(tf.reshape(outputsTF,[20,120]))
    Error = tf.reduce_mean(tf.square(outputsTF-outputdata))
    loss_op = loss_op + Error
    return loss_op,outputsTF


def batchSizeTrain():
    loss_op = 0.0
    lis = []
    for i in range(batch_size):
        inputdata = indata[i]
        outputdata = otdata[i]
        y = label[i]
        loss,outputsArray = lossFunction(inputdata,y,outputdata)
        lis.append(outputsArray)
        loss_op += loss
    return loss_op,lis

#loss_op = lossFunction()
loss_op,outputsArray = batchSizeTrain()
# Define loss and optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rata)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
#correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

from spe import getdata,gettestData
sepcindata,specotdata,speclabel,specsize = getdata()

def getTrainDataBatch():
    allindex= [i for i in range(0,specsize)]
    #print(len(allindex))
    index = np.random.permutation(allindex)
    index = index[0:batch_size]
    lisin = []
    lisot = []
    lislb = []
    for i in index:
        lisin.append(sepcindata[:,:,i])
        lisot.append(specotdata[:,:,i])
        lislb.append(speclabel[:,:,i])
    return np.asarray(lisin),np.asarray(lisot),np.asarray(lislb)

Tsepcindata,Tspecotdata,Tspeclabel,Tspecsize  = gettestData()
def getTestData(index):
    #allindex= [i for i in range(0,Tspecsize)]
    #print(len(allindex))
    #index = np.random.permutation(allindex)
    #index = index[0:batch_size]
    #index = allindex[index]
    lisin = []
    lisot = []
    lislb = []
    for i in index:
        lisin.append(Tsepcindata[:,:,i])
        lisot.append(Tspecotdata[:,:,i])
        lislb.append(Tspeclabel[:,:,i])
    return np.asarray(lisin),np.asarray(lisot),np.asarray(lislb)


if __name__ == "__main__":
    import os
    ckpt_dir = "./ckpt_dir"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    saver = tf.train.Saver()
    TRAIN = False
    if TRAIN:
        with tf.Session() as sess:
            sess.run(init)
            for i in range(0,training_iters):
                inLstm,otLstm,labelr = getTrainDataBatch()
                #print(inLstm.shape)
                #inLstm = tf.convert_to_tensor(inLstm)
                #otLstm = tf.convert_to_tensor(otLstm)
                #label = tf.convert_to_tensor(label)
                #sess.run(accuracy, feed_dict={X: test_data, Y: test_label})
                sess.run([train_op,outputsArray],feed_dict={indata:inLstm,otdata:otLstm,label:labelr})
                if i%100 == 0:
                    print(train_op,i)
            saver.save(sess,ckpt_dir+'/model.ckpt')
    else:
        with tf.Session() as sess:
            lis = []
            for i in range(0,int(Tspecsize/batch_size)):
                index = [batch_size*i + j for j in range(0,batch_size)]
                inLstm,otLstm,labelr = getTestData(index)
                model_file=tf.train.latest_checkpoint('ckpt_dir/')
                saver.restore(sess,model_file)
                val_loss,val_acc=sess.run([loss_op,outputsArray], feed_dict={indata:inLstm,otdata:otLstm,label:labelr})
                lis.append(val_acc)
            from utils import *
            dumpValue(lis,'spe.txt')










