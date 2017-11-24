
import utils
import gan
import config

import tensorflow as tf
import numpy as np
import os

def main(args):
  mnist = utils.read_data_sets(args.train_dir)
  config_proto = utils.get_config_proto()

  with tf.device('/gpu:0'):
    if not os.path.exists("../saves"):
      os.mkdir("../saves")
    sess = tf.Session(config=config_proto)
    model = gan.GAN(args, session)
    total_batch = mnist.train.num_examples // args.batch_size

    for epoch in range(1, args.nb_epochs + 1):
      for i in range(1, total_batch + 1):
        global_step = sess.run(model.global_step)
        x_batch, _ = mnist.train.next_batch(args.batch_size)
        noise = np.random.normal(size=[args.batch_size, args.noise_dim])

        D_loss = model.d_batch_fit(x_batch, noise)
        G_loss = model.g_batch_fit(noise)

        if i % args.log_period == 0:
          print "Epoch: ", '%02d' % epoch, "Batch: ", '%04d' % i, "D_loss: ", '%9.9f' % D_loss, "G_loss: ", '%9.9f' % G_loss

      if epoch % 50 == 0:
        print "- " * 50

      if epoch % args.save_period  == 0:
        if not os.path.exists("../saves/imgs"):
          os.mkdir("../saves/imgs")
        z = np.random.normal(size=[64, args.noise_dim])
        gen_images = np.reshape(model.generate(z), (64, 28, 28, 1))
        utils.save_images(gen_images, [8, 8], os.path.join(args.save_dir, "imgs/sample%s.jpg" % epoch))
    
if __name__ == '__main__':
  args = config.get_args()
  main(args)