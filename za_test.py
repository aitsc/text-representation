import tensorflow as tf
import numpy as np
# Build a graph.
a = tf.truncated_normal([1024, 100, 1], stddev=0.1)
# Launch the graph in a session.
sess = tf.Session()
# Evaluate the tensor `c`.
np.save('za_w.npy', sess.run(a))
print(sess.run(a))

