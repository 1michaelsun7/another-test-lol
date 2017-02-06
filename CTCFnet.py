import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imsave
import time
import tensorflow as tf
from utils import *

data_dir = "./maps.chr1/"
train_size = 2000
test_size = 500

class CTCFNet(object):
	def __init__(self, dirs, num_classes=2):

		self.num_classes = num_classes

		self.train_data = []
		self.train_labels = []
		self.test_data = []
		self.test_labels = []

		self.generate_training_set_and_labels(dirs)
		self.graph = tf.Graph()
		self.create_tf_graph()


	def generate_training_set_and_labels(self, dirs):
		class_num = 0
		for d in dirs:
			data_array = []
			with open(d, 'r') as f_pos:
				header = f_pos.readline()[1:]
				split_header = header.split("\t")
				self.x_dim, self.y_dim = map(int, split_header)
				for line in f_pos:
					splitline = line.split("\t")
					splitline = map(float, splitline)
					splitline = np.array(splitline)
					data_array.append(splitline)
			data_array = np.array(data_array)
			label_array = np.zeros((data_array.shape[0], self.num_classes))
			label_array[:, class_num] = 1

			self.train_data.append(data_array[:train_size])
			self.train_labels.append(label_array[:train_size])
			self.test_data.append(data_array[train_size:test_size + train_size])
			self.test_labels.append(label_array[train_size:test_size + train_size])

			class_num += 1

		self.train_data = np.concatenate(self.train_data)
		self.train_labels = np.concatenate(self.train_labels)
		self.test_data = np.concatenate(self.test_data)
		self.test_labels = np.concatenate(self.test_labels)

		self.train_data, self.train_labels = shuffle_in_unison(self.train_data, self.train_labels)
		self.test_data, self.test_labels = shuffle_in_unison(self.test_data, self.test_labels)


	def create_tf_graph(self):
		batch_size = 50
		learning_rate = 0.01
		conv1_filter_size = 5
		conv1_depth = 8
		conv2_filter_size = 5
		conv2_depth = 16
		fc_num_hidden = 64

		keep_prob = 0.8 # set to < 1.0 to apply dropout, 1.0 to remove
		weight_penalty = 0.01 # set to > 0.0 to apply weight penalty, 0.0 to remove

		# train_cutoff = int(math.floor(0.9*self.data.shape[0]))
		# train_set, train_label = self.data[:train_cutoff, :], self.labels[:train_cutoff,]
		# val_set, val_label = self.data[train_cutoff:, :], self.labels[train_cutoff:]
		# test_set, test_labels = shuffle_in_unison(self.data, self.labels)
		# test_set = test_set[:256]
		# test_labels = test_labels[:256]

		with self.graph.as_default():
			train_x = tf.placeholder(tf.float32, shape=(None, self.x_dim*self.y_dim))
			train_y = tf.placeholder(tf.float32, shape=(None, self.num_classes))

			x_image = tf.reshape(train_x, [-1, self.x_dim, self.y_dim, 1])

			W_conv1 = weight_variable([conv1_filter_size, conv1_filter_size, 1, conv1_depth])
			b_conv1 = bias_variable([conv1_depth])

			W_conv1_min = tf.reduce_min(W_conv1)
			W_conv1_max = tf.reduce_max(W_conv1)
			W_conv1_0_to_1 = (W_conv1 - W_conv1_min)/(W_conv1_max - W_conv1_min)
			W_conv1_transpose = tf.transpose(W_conv1_0_to_1, [3, 0, 1, 2])
			tf.summary.image("conv1/filters", W_conv1_transpose)

			h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
			h_pool1 = max_pool(h_conv1)

			conv1_feat_map_x = int(math.ceil(float(self.x_dim) / 2)) # stride
			conv1_feat_map_y = int(math.ceil(float(self.y_dim) / 2)) # stride

			W_conv2 = weight_variable([conv2_filter_size, conv2_filter_size, conv1_depth, conv2_depth])
			b_conv2 = bias_variable([conv2_depth])

			h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
			h_pool2 = max_pool(h_conv2)

			conv2_feat_map_x = int(math.ceil(float(conv1_feat_map_x) / 2)) # stride
			conv2_feat_map_y = int(math.ceil(float(conv1_feat_map_y) / 2)) # stride

			W_fc1 = weight_variable([conv2_feat_map_x * conv2_feat_map_y * conv2_depth, fc_num_hidden])
			b_fc1 = bias_variable([fc_num_hidden])

			h_pool2_flat = tf.reshape(h_pool2, [-1, conv2_feat_map_x * conv2_feat_map_y * conv2_depth])
			h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

			h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

			W_fc2 = weight_variable([fc_num_hidden, self.num_classes])
			b_fc2 = bias_variable([self.num_classes])

			y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

			gradients = tf.gradients(y_conv, train_x)
			gradients = tf.gather(tf.reshape(gradients, [-1, self.x_dim, self.y_dim, 1]),0)

			cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, train_y))
			loss = cross_entropy + weight_decay_penalty([W_fc1, W_fc2])*weight_penalty
			train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
			correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(train_y,1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

			def train_model(num_epochs):
				batches_per_epoch = int(len(self.train_data)/batch_size)
				num_steps = int(num_epochs * batches_per_epoch)
				with tf.Session(graph=self.graph) as sess:
					summary_writer = tf.summary.FileWriter('/tmp/logs', sess.graph)
					merged = tf.summary.merge_all()

					tf.global_variables_initializer().run()
					print 'Initializing variables...'

					for step in xrange(num_steps):
						offset = (step * batch_size) % (self.train_data.shape[0] - batch_size)
						batch_x = self.train_data[offset:(offset + batch_size), :]
						batch_y = self.train_labels[offset:(offset + batch_size)]

						feed_dict = {train_x: batch_x, train_y: batch_y}
						_, batch_loss, batch_preds, batch_acc = sess.run([train_step, loss, correct_prediction, accuracy], feed_dict=feed_dict)

						if (step % 100 == 0):
							print ''
							print('Batch loss at step %d: %f' % (step, batch_loss))
							print('Training accuracy: %f' % batch_acc)

					test_summary, test_acc, test_grads = sess.run([merged, accuracy, gradients], feed_dict={train_x: self.test_data, train_y: self.test_labels})
					summary_writer.add_summary(test_summary, step)
					print('Test accuracy: %f' % test_acc)

					summary_writer.close()
				print sess.run(accuracy, feed_dict={train_x: self.test_data, train_y: self.test_labels})

					# for ii in xrange(5):
					# 	image_for_conv = self.train_data[ii:ii+1]
					# 	img_filter1, img_filter2 = sess.run([h_conv1,h_conv2], feed_dict={train_x: image_for_conv})
					# 	plt.imsave("orig_image" + str(ii), np.reshape(image_for_conv, [self.x_dim, self.y_dim]), cmap="gray")
					# 	# plt.imshow(np.reshape(image_for_conv, [self.x_dim, self.y_dim]), interpolation="nearest")

					# 	plot_filter(img_filter1, layer_name="h_conv1_" + str(ii))
					# 	plot_filter(img_filter2, layer_name="h_conv2_" + str(ii))

			self.train_function = train_model

	
if __name__ == '__main__':
	CTCF_fpath = data_dir + 'DNase.CTCF.chr1.data.txt'
	ETS1_fpath = data_dir + 'DNase.ETS1.chr1.data.txt'
	conv_net = CTCFNet([ETS1_fpath, CTCF_fpath])

	start = time.time()
	conv_net.train_function(1)
	print("Training time: ", time.time() - start)