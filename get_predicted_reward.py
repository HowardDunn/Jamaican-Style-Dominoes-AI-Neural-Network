import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp
import numpy as np

print("Loading model")
saver = None 
try:
    saver = tf.train.import_meta_graph("{}.meta".format("./checkpoints/model_final.ckpt"))
except:
    pass


def predict(X):
    global saver
    with tf.Session() as sess:
        
        saver.restore(sess,"./checkpoints/model_final.ckpt")
        #print(sess.run('x:0'))
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        prediction = graph.get_tensor_by_name("prediction:0")
        #print(sess.run(op_to_restore,feed_dict))
        nx = np.matrix(X)
        preds = prediction.eval(feed_dict = {x:nx})
    return preds[0][0]


def get_predicted_reward(game_state):

    return predict(game_state)