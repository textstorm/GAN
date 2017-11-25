
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
    self.sess = sess
    self.max_grad_norm = args.max_grad_norm
    self.learning_rate = args.learning_rate

    self.add_placeholder()

    self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
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

  def _build_forward(self):
    with tf.name_scope("model"):
      with tf.variable_scope("generator"):
        self.G, self.g_var_list = self.generator(self.noise)
      with tf.variable_scope("discrininator"):
        self.D_real, self.d_var_list = self.discriminator(self.x_images)
      with tf.variable_scope("discrininator", reuse=True):
        self.D_fake, _ = self.discriminator(self.G)
  
  def _build_loss(self):
    with tf.name_scope("Loss"):
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
    g_w1 = weight_variable([self.noise_dim, self.g_h1_dim], name="W1_G")
    g_b1 = bias_variable([self.g_h1_dim], name="b1_G")
    g_h1 = tf.nn.relu(tf.matmul(noise, g_w1) + g_b1)
    g_w2 = weight_variable([self.g_h1_dim, self.input_dim], name="W2_G")
    g_b2 = bias_variable([self.input_dim], name="b2_G")
    g_h2 = tf.nn.sigmoid(tf.matmul(g_h1, g_w2) + g_b2)

    var_list = [g_w1, g_w2, g_b1, g_b2]

    return g_h2, var_list

  def discriminator(self, x): 
    d_w1 = weight_variable([self.input_dim, self.d_h1_dim], name="W1_D")
    d_b1 = bias_variable([self.d_h1_dim], name="b1_D")
    d_h1 = tf.nn.relu(tf.matmul(x, d_w1) + d_b1)
    d_w2 = weight_variable([self.d_h1_dim, 1], name="W2_D")
    d_b2 = bias_variable([1], name="b2_D")
    d_l2 = tf.matmul(d_h1, d_w2) + d_b2
    var_list = [d_w1, d_w2, d_b1, d_b2]

    return d_l2, var_list

  def generate(self, noise):
    feed_dict = {self.noise: noise}
    return self.sess.run(self.G, feed_dict=feed_dict)