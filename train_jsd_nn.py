import tensorflow as tf
import matplotlib.pyplot as plt
from game_state_capture import load_data
import random
import numpy as np
import socket,os,sys
from threading import Thread
import random
#from game_loop import GameLoop
from board_memory import *
from domino import *
from user import *
from game_state_capture import load_data,save_actions
from get_predicted_reward import open_tf_session, close_tf_session
import datetime

'''
gameType='cutthroat'
gameloop = GameLoop(type=gameType,use_nn=True)


def PlayGame(num_games=5):
    open_tf_session()
    print ("Playing ",num_games, "to get win percentage")
    load_data("dummy.csv")
    global gameloop
    total_wins = 0
    average_opponent_wins = 0
    total_games = 0
    global gameType
    for i in range(0,num_games):
        wins,average_opponent, total = gameloop.run()
        total_wins += wins
        total_games += total
        average_opponent_wins += average_opponent
        gameloop = GameLoop(type=gameType,use_nn=True)
        
    file = open('metrics.txt','a')
    file.write(str(total_wins/total_games) + ',' + str(average_opponent_wins/total_games) + ',' + str(total_wins) + ',' + str(average_opponent_wins) + ',' + str(total_games) + ','  + (str(datetime.datetime.now())) + '\n')
    file.close()

    print("Win percentage: ", (total_wins/total_games))
    print("Opponent win percentage: ", (average_opponent_wins/total_games))
    print("Total wins = ",total_wins, 'Average Opponent wins = ', average_opponent_wins,'Total games = ', total_games)
    save_actions()
    close_tf_session()

    return (total_wins/total_games)
''' 

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
            

def massage_data(actions_and_rewards,ratio=0.1):
    
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
    n_features = 141

    # Hidden Layers
    hidden_layers = [50,50]

    # Output nodes
    n_classes = 1

    # Relieves stress and does 100 features at a time
    batch_size = 400
    
    trainX, trainY, testX,testY = get_data()

    trainX = np.matrix(trainX)
    trainY = np.matrix(trainY).T
    testX = np.matrix(testX)
    testY = np.matrix(testY).T
   

    x = tf.placeholder('float32',[None, n_features],name="x",)
    y = tf.placeholder('float32',name="y")
    tf.add_to_collection("x", x)
    tf.add_to_collection("x", y)
    train_neural_network(x,y,trainX,trainY,testX,testY,n_features,hidden_layers,n_classes,epochs=5,batch_size=batch_size)

def train_jsd_ai_incremental(actions_and_rewards,saver,model=5,restore_session=False,session=None):
    '''
    input data -> weight values -> run through hidden layer 1 (activation function) -> weights ->
     run through hidden layer 2 (activation function) -> weights -> output layer
    compare output with intended output -> calculate cost -> minimize cost (Adam, SGD, AdaGrad)
    using   back propagation

    feed forward + backprop = epoch
    :return:
    '''
    print("Training")
    print()
    # Input Layer
    n_features = 141

    # Hidden Layers
    hidden_layers = [50,50]

    # Output nodes
    n_classes = 1

    # Relieves stress and does 100 features at a time
    batch_size = 400
    
    trainX, trainY, testX,testY = massage_data(actions_and_rewards)

    trainX = np.matrix(trainX)
    trainY = np.matrix(trainY).T
    testX = np.matrix(testX)
    testY = np.matrix(testY).T
   

    x = tf.placeholder('float32',[None, n_features],name="x",)
    y = tf.placeholder('float32',name="y")
    tf.add_to_collection("x", x)
    tf.add_to_collection("x", y)
    train_neural_network(x,y,trainX,trainY,testX,testY,n_features,hidden_layers,n_classes,epochs=5,batch_size=batch_size,saver=saver,model=model,restore_session=restore_session,session=session)


def jsd_nn_model(data, n_features,hidden_layer_nodes,n_classes):

    hidden_layers = [None]*len(hidden_layer_nodes)

    assert(len(hidden_layer_nodes))
    prev_n_nodes = n_features
    count = 1
    
    hidden_layer_1 = {
                        'weights': tf.Variable(name='hidden_layer_1',initial_value = tf.random_normal([n_features,hidden_layer_nodes[0]])),
                        'biases':  tf.Variable(name='hidden_layer_bias_1', initial_value=tf.random_normal([hidden_layer_nodes[0]]))
                        }
    hidden_layer_2 = {
                        'weights': tf.Variable(name='hidden_layer_2',initial_value = tf.random_normal([hidden_layer_nodes[0],hidden_layer_nodes[1]])),
                        'biases':  tf.Variable(name='hidden_layer_2', initial_value=tf.random_normal([hidden_layer_nodes[1]]))
                        }
    output_layer = {
        'weights': tf.Variable(tf.random_normal([hidden_layer_nodes[1], n_classes])),
        'biases': tf.Variable(tf.random_normal([n_classes]))
    }

    layer_1 =   tf.add(tf.matmul(data,hidden_layer_1['weights']), hidden_layer_1['biases'])
    layer_1 =   tf.nn.relu(layer_1)
    layer_2 =   tf.add(tf.matmul(layer_1,hidden_layer_2['weights']), hidden_layer_2['biases'])
    layer_2 =   tf.nn.relu(layer_2)   

    output = tf.matmul(layer_2,output_layer['weights']) + output_layer['biases']
    tf.add_to_collection("hidden_layer_1", hidden_layer_1['weights'])
    tf.add_to_collection("hidden_layer_bias_1", hidden_layer_1['biases'])
    tf.add_to_collection("hidden_layer_2", hidden_layer_2['weights'])
    tf.add_to_collection("hidden_layer_bias_2", hidden_layer_2['biases'])
    return output


def neural_network_model(data, n_features,hidden_layer_nodes,n_classes):

    hidden_layers = [None]*len(hidden_layer_nodes)

    assert(len(hidden_layer_nodes))
    prev_n_nodes = n_features
    count = 1

    for i,n_nodes in enumerate(hidden_layer_nodes):
        
       
        name = 'hidden_layer_' + str(count)
        bname = 'hidden_layer_bias_' + str(count)  
        hidden_layers[i] = {
                        'weights': tf.Variable(name=name,initial_value = tf.random_normal([prev_n_nodes,n_nodes])),
                        'biases':  tf.Variable(name=bname, initial_value=tf.random_normal([n_nodes]))
                        }
        prev_n_nodes = n_nodes

        count += 1

    output_layer = {
        'weights': tf.Variable(tf.random_normal([prev_n_nodes, n_classes])),
        'biases': tf.Variable(tf.random_normal([n_classes]))
    }

    layers = [None]*len(hidden_layer_nodes)
    prev_layer = data
    for i,hidden_layer in enumerate(hidden_layers):

        layers[i] = tf.add(tf.matmul(prev_layer,hidden_layer['weights']), hidden_layer['biases'])
        layers[i] = tf.nn.relu(layers[i])
        prev_layer = layers[i]
       

    output = tf.matmul(prev_layer,output_layer['weights']) + output_layer['biases']
    
    return output


def train_neural_network(x,y,trainX, trainY, testX, testY,n_features,hidden_layers,num_outputs,
                                learning_rate=0.01,epochs=5, batch_size=100,saver=None,model= 5,restore_session=True, session=None):

    

    n_samples = len(trainX)
    graph = tf.get_default_graph()
    if session == None:
        prediction_model1 = jsd_nn_model(x,n_features,hidden_layers,num_outputs)
        prediction_model1 = tf.identity(prediction_model1, name="prediction_model1")
    
        prediction_model2 = jsd_nn_model(x,n_features,hidden_layers,num_outputs)
        prediction_model2 = tf.identity(prediction_model2, name="prediction_model2")
    
        prediction_model3 = jsd_nn_model(x,n_features,hidden_layers,num_outputs)
        prediction_model3 = tf.identity(prediction_model3, name="prediction_model3")
    
        prediction_model4 = jsd_nn_model(x,n_features,hidden_layers,num_outputs)
        prediction_model4 = tf.identity(prediction_model4, name="prediction_model4")
        cost1 = tf.reduce_mean(tf.abs(y - prediction_model1),name="cost1")  #tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
        cost2 = tf.reduce_mean(tf.abs(y - prediction_model2),name="cost2") 
        cost3 = tf.reduce_mean(tf.abs(y - prediction_model3),name="cost3") 
        cost4 = tf.reduce_mean(tf.abs(y - prediction_model4),name="cost4") 

        optimizer1 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost1)
        optimizer2 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost2)
        optimizer3 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost3)
        optimizer4 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost4)

        print("adding to graph collection")
        tf.add_to_collection("prediction_model1", prediction_model1)
        tf.add_to_collection("prediction_model2", prediction_model2)
        tf.add_to_collection("prediction_model3", prediction_model3)
        tf.add_to_collection("prediction_model4", prediction_model4)

        tf.add_to_collection("cost1", cost1)
        tf.add_to_collection("cost2", cost2)
        tf.add_to_collection("cost3", cost3)
        tf.add_to_collection("cost4", cost4)


      
    else:
        prediction_model1 = graph.get_tensor_by_name("prediction_model1:0")
        prediction_model2 = graph.get_tensor_by_name("prediction_model2:0")
        prediction_model3 = graph.get_tensor_by_name("prediction_model3:0")
        prediction_model4 = graph.get_tensor_by_name("prediction_model4:0")

        x = graph.get_tensor_by_name("x:0")
        y = graph.get_tensor_by_name("y:0")

        cost1 = graph.get_tensor_by_name("cost1:0")
        cost2 = graph.get_tensor_by_name("cost2:0")
        cost3 = graph.get_tensor_by_name("cost3:0")
        cost4 = graph.get_tensor_by_name("cost4:0")
        #op = graph.get_operations()
        #for o in op:
        #    print("Op: ", o.name)
        optimizer1 = tf.get_collection("Adam")
        optimizer2 = tf.get_collection("Adam_1")
        optimizer3 = tf.get_collection("Adam_2")
        optimizer4 = tf.get_collection("Adam_3")
        #optimizer2 = graph.get_tensor_by_name("Adam_1:0")
        #optimizer3 = graph.get_tensor_by_name("Adam_2:0")
       # optimizer4 = graph.get_tensor_by_name("Adam_3:0")

       

       
        
    

    
    #tf.get_default_graph().finalize()
    
    if saver == None:
        saver = tf.train.Saver()
        restore_session = False


        
    losses = []
    iterations = []
    win_percentages = []
    if session == None:
        session = tf.Session()
    
    sess = session 
    writer = tf.summary.FileWriter("output", sess.graph)
    if not restore_session:
        sess.run(tf.global_variables_initializer())

    num_batches = int(n_samples/batch_size) + 1
    print("Num batches: ", num_batches, " Num samples: ", n_samples)
        
    file = open('loss.txt','w')
    file.write('')
    file.close()
    if restore_session:
        try:
                
            #saver = tf.train.import_meta_graph("{}.meta".format("./checkpoints/model.ckpt"))
            #checkpoint = tf.train.get_checkpoint_state("./checkpoints/model.ckpt")
            saver.restore(sess, "./checkpoints/model.ckpt")
            print("Model restored.")
        except:
            print("Cannot restore checkpoint")
        
    for j in range(0, 1):
        #saver.restore(sess, "./checkpoints/model.ckpt")
        for epoch in range(0,epochs):
            epoch_loss = 0

            for i in range(0,num_batches):
                #epoch_x,epoch_y = mnist.train.next_batch(batch_size)
                epoch_x = np.array(trainX[i : (i+1)*batch_size])
                epoch_y = np.array(trainY[i:(i+1)*batch_size])
                if model == 1:
                    _,c = sess.run([optimizer1,cost1],feed_dict={x: epoch_x, y: epoch_y})
                elif model == 2:
                    _,c = sess.run([optimizer2,cost2],feed_dict={x: epoch_x, y: epoch_y})
                elif model == 3:
                    _,c = sess.run([optimizer3,cost3],feed_dict={x: epoch_x, y: epoch_y})
                elif model == 4:
                    _,c = sess.run([optimizer4,cost4],feed_dict={x: epoch_x, y: epoch_y})
                else:
                    _,c1 = sess.run([optimizer1,cost1],feed_dict={x: epoch_x, y: epoch_y})
                    _,c2 = sess.run([optimizer2,cost2],feed_dict={x: epoch_x, y: epoch_y})
                    _,c3 = sess.run([optimizer3,cost3],feed_dict={x: epoch_x, y: epoch_y})
                    _,c4 = sess.run([optimizer4,cost4],feed_dict={x: epoch_x, y: epoch_y})
                    c = c1 + c2+ c3 + cost4
                    c /= 4.0
                epoch_loss += c
                print("Completed Batch: ", (i+1), " out of: ", num_batches)

            print("Epoch: ", epoch+1, ' completed out of', epochs, 'loss: ', epoch_loss)
            losses.append(epoch_loss)
            iterations.append(epoch)
            file = open('loss.txt','a')
            file.write(str(epoch) + ',' + str(epoch_loss) + '\n')
            file.close()
            save_path = saver.save(sess, "./checkpoints/model.ckpt",write_meta_graph=False)
            print("Model saved in path: %s" % save_path)
        
        with sess.as_default():
            preds1 = prediction_model1.eval(feed_dict = {x:testX})
            preds2 = prediction_model2.eval(feed_dict = {x:testX})
            preds3 = prediction_model3.eval(feed_dict = {x:testX})
            preds4 = prediction_model4.eval(feed_dict = {x:testX})

            test_cost1 = cost1.eval(feed_dict = {y:testY,prediction_model1:preds1})
            test_cost2 = cost2.eval(feed_dict = {y:testY,prediction_model2:preds2})
            test_cost3 = cost3.eval(feed_dict = {y:testY,prediction_model3:preds3})
            test_cost4 = cost4.eval(feed_dict = {y:testY,prediction_model4:preds4})

            print("Test Loss1: ", test_cost1)
            print("Test Loss2: ", test_cost2)
            print("Test Loss3: ", test_cost3)
            print("Test Loss4: ", test_cost4)
        save_path = saver.save(sess, "./checkpoints/model.ckpt")
        print("Model saved in path: %s" % save_path)
      
        save_path = saver.save(sess, "./checkpoints/model_final.ckpt")
        
        print("Model Final saved in path: %s" % save_path)
            #win_percentage = PlayGame()
            #win_percentages.append(win_percentage)
        
            
            
            

        writer.close()
    

#neural_network_model(input_data,784, [500,500,500,],10)

#train_jsd_ai()