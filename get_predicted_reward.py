import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp
import numpy as np

print("Loading model")
tf.logging.set_verbosity(tf.logging.ERROR)
saver = None 
sess = None
session_restored = False
def open_tf_session():
	global sess
	global saver
	sess = tf.Session()
	saver = tf.train.import_meta_graph("{}.meta".format("./checkpoints/model.ckpt"))
	total_parameters = 0
	for variable in tf.trainable_variables():  
		local_parameters=1
		shape = variable.get_shape()
		print("Name: ", variable.name)
		for i in shape:
			local_parameters*=i.value
		total_parameters+=local_parameters
	print(total_parameters) 
	print("Finished importing meta graph")
def close_tf_session():
	sess.close()

def get_saver():
	return saver

def get_session():
	return sess

def predict(X,model=5):
	global saver
	global session_restored

	if not session_restored:
		saver.restore(sess,"./checkpoints/model.ckpt")
		session_restored = True
	#print(sess.run('x:0'))
	graph = tf.get_default_graph()
	x = graph.get_tensor_by_name("x:0")
	prediction_model1 = graph.get_tensor_by_name("prediction_model1:0")
	prediction_model2 = graph.get_tensor_by_name("prediction_model2:0")
	prediction_model3 = graph.get_tensor_by_name("prediction_model3:0")
	prediction_model4 = graph.get_tensor_by_name("prediction_model4:0")
	#print(sess.run(op_to_restore,feed_dict))

	nx = np.matrix(X)
	preds1 = prediction_model1.eval(session=sess,feed_dict = {x:nx})
	preds2 = prediction_model2.eval(session=sess,feed_dict = {x:nx})
	preds3 = prediction_model3.eval(session=sess,feed_dict = {x:nx})
	preds4 = prediction_model4.eval(session=sess,feed_dict = {x:nx})
	if model == 1:
		preds = preds1[0][0]
	if model == 2:
		preds = preds2[0][0]
	if model == 3:
		preds = preds3[0][0]
	if model == 4:
		preds = preds4[0][0]
	if model == 5:
		preds = preds1[0][0] + preds2[0][0] + preds3[0][0] + preds4[0][0]
		preds = preds/4.0

	#print("Expected Reward 1: ", preds1[0][0],  "Expected Reward 2: ", preds2[0][0], "Expected Reward 3: ", preds3[0][0], "Expected Reward 4: ", preds4[0][0])
	
	return preds


def get_predicted_reward(game_state,model=5):

    return predict(game_state,model=model)
