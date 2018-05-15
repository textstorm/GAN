
import tensorflow as tf

def weight_variable(shape, name, initializer=None):
  initializer = tf.truncated_normal_initializer(stddev=0.1)
  if initializer:
    initializer = initializer
  return tf.get_variable(shape=shape, initializer=initializer, name=name)

def bias_variable(shape, name, initializer=None):
  initializer = tf.constant_initializer(0.)
  if initializer:
    initializer = initializer
  return tf.get_variable(shape=shape, initializer=initializer, name=name)

class GAN(object):
  def __init__(self, args, sess, name='gan'):
    self.input_dim = args.input_dim
    self.noise_dim = args.noise_dim
    self.g_h1_dim = args.g_h1_dim
    self.g_h2_dim = args.g_h2_dim
    self.d_h1_dim = args.d_h1_dim
    self.d_h2_dim = args.d_h2_dim

    self.sess = sess
    self.max_grad_norm = args.max_grad_norm
    self.learning_rate = args.learning_rate
    self.keep_prob = args.keep_prob

    self.add_placeholder()

    self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
    self.global_step = tf.Variable(0, trainable=False)
    
    self._build_graph()
    self._build_loss()
    self._build_train()

    init_op = tf.global_variables_initializer()
    self.sess.run(init_op)

  def add_placeholder(self):
    with tf.name_scope("Data"):
      self.x_images = tf.placeholder(tf.float32, [None, self.input_dim])
      self.noise = tf.placeholder(tf.float32, [None, self.noise_dim])

  def _build_graph(self):
    with tf.variable_scope("generator"):
      self.G, self.g_var_list = self.generator(self.noise)

    with tf.variable_scope("discriminator"):
      self.D_real, self.d_var_list = self.discriminator(self.x_images)
    with tf.variable_scope("discriminator", reuse=True):
      self.D_fake, _ = self.discriminator(self.G)
  
  def _build_loss(self):
    with tf.name_scope("loss"):
      D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                   logits=self.D_real, labels=tf.ones_like(self.D_real)))
      D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                   logits=self.D_fake, labels=tf.zeros_like(self.D_fake)))
      G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                              logits=self.D_fake, labels=tf.ones_like(self.D_fake)))

      self.D_loss =  (D_loss_real + D_loss_fake)
      self.G_loss =  G_loss

  def _build_train(self):
    with tf.name_scope("train"):
      d_grads_and_vars = self.optimizer.compute_gradients(self.D_loss, var_list=self.d_var_list)
      d_grads_and_vars = [(tf.clip_by_norm(g, self.max_grad_norm), v) for g, v in d_grads_and_vars]
      g_grads_and_vars = self.optimizer.compute_gradients(self.G_loss, var_list=self.g_var_list)
      g_grads_and_vars = [(tf.clip_by_norm(g, self.max_grad_norm), v) for g, v in g_grads_and_vars]
      self.train_op_d = self.optimizer.apply_gradients(d_grads_and_vars, global_step=self.global_step)
      self.train_op_g = self.optimizer.apply_gradients(g_grads_and_vars, global_step=self.global_step)

  def d_batch_fit(self, x_images, noise):
    feed_dict = {self.x_images: x_images, self.noise:noise}
    loss, _ = self.sess.run([self.D_loss, self.train_op_d], feed_dict=feed_dict)
    return loss

  def g_batch_fit(self, noise):
    feed_dict = {self.noise: noise}
    feed_dict[self.noise] = noise
    loss, _ = self.sess.run([self.G_loss, self.train_op_g], feed_dict=feed_dict)
    return loss

  def generator(self, noise):
    x = noise
    x = tf.layers.dense(x, self.g_h1_dim, activation=tf.nn.relu, name='g_layer1')
    x = tf.layers.dense(x, self.g_h2_dim, activation=tf.nn.relu, name='g_layer2')
    x = tf.layers.dense(x, self.input_dim, name='g_layer3')
    x = tf.nn.sigmoid(x)
    tvars = self.trainable_vars('generator')
    print tvars
    return x, tvars

  def discriminator(self, x):
    x = tf.layers.dense(x, self.d_h1_dim, activation=tf.nn.relu, name='d_layer1')
    x = tf.nn.dropout(x, keep_prob=self.keep_prob)
    x = tf.layers.dense(x, self.d_h2_dim, activation=tf.nn.relu, name='d_layer2')
    x = tf.nn.dropout(x, keep_prob=self.keep_prob)
    x = tf.layers.dense(x, 1, activation=None, name='d_layer3')
    tvars = self.trainable_vars('discriminator')
    print tvars
    return x, tvars

  def generate(self, noise):
    feed_dict = {self.noise: noise}
    return self.sess.run(self.G, feed_dict=feed_dict)

  def trainable_vars(self, scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)