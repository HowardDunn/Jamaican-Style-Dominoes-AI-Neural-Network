import tensorflow as tf
from game_state_capture import load_data
import random
import numpy as np

def  basic_multiply():
    x1 = tf.constant(5)
    x2 = tf.constant(6)

    result = tf.multiply(x1,x2) # multiplies

    with tf.Session() as sess:
        print(sess.run(result))

def get_data(ratio=0.1):
    assert(ratio < 1)
    actions_and_rewards = load_data("dummy.csv")
    
    totalX = []
    totalY = []

    for action_str in actions_and_rewards:
        
        action = action_str.split(',')
        totalX.append(action)
        totalY.append(actions_and_rewards[action_str][2])
    
    trainX = []
    trainY = []
    testX = []
    testY = []

    threshold = 100*ratio
    for i in range(0,len(totalX)):
        
        ran_num = random.randint(0,100)
        
        if ran_num < threshold:
            testX.append(totalX[i])
            testY.append(totalY[i])
        else:
            trainX.append(totalX[i])
            trainY.append(totalY[i])

    print("TrainX: ",len(trainX))
    print("TrainY: ",len(trainY))
    print("TestX: ",len(testX))
    print("TestY: ",len(testY))
    return trainX,trainY,testX,testY
            

    


def train_jsd_ai():
    '''
    input data -> weight values -> run through hidden layer 1 (activation function) -> weights ->
     run through hidden layer 2 (activation function) -> weights -> output layer
    compare output with intended output -> calculate cost -> minimize cost (Adam, SGD, AdaGrad)
    using   back propagation

    feed forward + backprop = epoch
    :return:
    '''

    # Input Layer
    n_features = 140

    # Hidden Layers
    hidden_layers = [280,140]

    # Output nodes
    n_classes = 1

    # Relieves stress and does 100 features at a time
    batch_size = 100
    
    trainX, trainY, testX,testY = get_data()

    trainX = np.matrix(trainX)
    trainY = np.matrix(trainY).T
    testX = np.matrix(testX)
    testY = np.matrix(testY).T
   

    x = tf.placeholder('float32',[None, n_features])
    y = tf.placeholder('float32',)
    train_neural_network(x,y,trainX,trainY,testX,testY,n_features,hidden_layers,n_classes,batch_size=batch_size)


def neural_network_model(data, n_features,hidden_layer_nodes,n_classes):

    hidden_layers = []

    assert(len(hidden_layer_nodes))
    prev_n_nodes = n_features
    count = 1

    for n_nodes in hidden_layer_nodes:
        print("Iters:",count)
        count += 1
        print("N_Nodes:",n_nodes)
        hidden_layer = {
                        'weights': tf.Variable(tf.random_normal([prev_n_nodes,n_nodes])),
                        'biases':  tf.Variable(tf.random_normal([n_nodes]))
                        }
        prev_n_nodes = n_nodes

        hidden_layers.append(hidden_layer)

    output_layer = {
        'weights': tf.Variable(tf.random_normal([prev_n_nodes, n_classes])),
        'biases': tf.Variable(tf.random_normal([n_classes]))
    }

    layers = []
    prev_layer = data
    for hidden_layer in hidden_layers:

        layer = tf.add(tf.matmul(prev_layer,hidden_layer['weights']), hidden_layer['biases'])
        layer = tf.nn.relu(layer)
        prev_layer = layer
        layers.append(layer)

    output = tf.matmul(prev_layer,output_layer['weights']) + output_layer['biases']
    return output


def train_neural_network(x,y,trainX, trainY, testX, testY,n_features,hidden_layers,num_outputs,
                                learning_rate=0.1,epochs=20, batch_size=100,):

    n_samples = len(trainX)
    prediction = neural_network_model(x,n_features,hidden_layers,num_outputs)
    
    cost = tf.reduce_mean(tf.square(y - prediction))  #tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    with tf.Session() as sess:
        writer = tf.summary.FileWriter("output", sess.graph)
        sess.run(tf.global_variables_initializer())

        for epoch in range(0,epochs):
            epoch_loss = 0

            for i in range(0,int(n_samples/batch_size)):
                #epoch_x,epoch_y = mnist.train.next_batch(batch_size)
                epoch_x = np.array(trainX[i : (i+1)*batch_size])
                epoch_y = np.array(trainY[i:(i+1)*batch_size])
                
                _,c = sess.run([optimizer,cost],feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print("Epoch: ", epoch, ' completed out of', epochs, 'loss: ', epoch_loss)

      
        difference = tf.abs(y-prediction)
        average = tf.abs(tf.scalar_mul(0.5,y+prediction))
        power_val = tf.divide(difference,average)
        correct = tf.reduce_mean(tf.exp(tf.negative(power_val)))
        preds = prediction.eval(feed_dict = {x:testX})
        for i in range (0, len(preds)):
            print("Predicted:" , preds[i],"Actual: ",testY[i])
        

        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        writer.close()
        print('Accuracy:', accuracy.eval({x:testX,y:testY}))

#neural_network_model(input_data,784, [500,500,500,],10)

train_jsd_ai()