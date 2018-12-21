import numpy as np
import tensorflow as tf

seq_len = 10
state_size = 10
batch_size = 32
init_space = 20

def gen_data(x0):
	c = np.random.rand()
	seq = np.zeros(seq_len)
	y = 0
	for i in range(seq_len):
		seq[i] = x0+(i+1)*c
		y += seq[i]
	return seq,y

def gen_batch():
	batch_x = []
	batch_y = []
	for i in range(batch_size):
		x0 = np.random.randint(init_space)
		x_,y_ = gen_data(x0)
		batch_x.append(x_)
		batch_y.append(y_)
	return batch_x, batch_y

# Placeholders
X = tf.placeholder(tf.float32, [None,1])
Y = tf.placeholder(tf.float32, [None,1])
# State
h = tf.placeholder(tf.float32, [None,state_size])

# State+input
x_h = tf.concat([X,h],1)

# First layer
W0 = tf.Variable(np.random.rand(state_size+1,state_size),dtype=tf.float32)
b0 = tf.Variable(np.zeros((1,state_size)),dtype=tf.float32)
O0 = tf.nn.relu(tf.matmul(x_h,W0)+b0)

# Second layer
W1 = tf.Variable(np.random.rand(state_size,1),dtype=tf.float32)
b1 = tf.Variable(np.zeros((1,1)),dtype=tf.float32)

# Output
y_ = tf.nn.relu(tf.matmul(O0,W1)+b1)

# Loss function
loss = tf.losses.mean_squared_error(Y,y_)
# Training
train = tf.train.AdamOptimizer(0.001).minimize(loss)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(2000):
		batch_x, batch_y = gen_batch()
		batch_x = np.array(batch_x)
		batch_y = np.array(batch_y).reshape((-1,1))
		states = np.zeros((batch_size,state_size))
		for j in range(seq_len-1):
			bx = batch_x[:,j].reshape((-1,1))
			states = 0.9*states+0.1*sess.run(O0,feed_dict={X:bx,h:states})
		bx = batch_x[:,seq_len-1].reshape((-1,1))
		states,loss_,train_ = sess.run([O0,loss,train],feed_dict={X:bx,Y:batch_y,h:states})
		print(loss_)

