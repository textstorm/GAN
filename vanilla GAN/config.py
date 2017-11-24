
import argparse

def get_args():

  parser = argparse.ArgumentParser()

  parser.add_argument('--random_seed', type=int, default=827, help='Random seed')

  #data file
  parser.add_argument('--train_dir', type=str, default='/root/textstorm/GAN/vanilla GAN/data/')
  parser.add_argument('--log_dir', type=str, default='/root/textstorm/GAN/vanilla GAN/logs/')
  parser.add_argument('--save_dir', type=str, default='/root/textstorm/GAN/vanilla GAN/saves/')
  parser.add_argument('--nb_classes', type=int, default=10)

  #model details
  parser.add_argument('--noise_dim', type=int, default=100, help='The input dims of generator')
  parser.add_argument('--input_dim', type=int, default=784, help='Dimension of input data')
  parser.add_argument('--g_h1_dim', type=int, default=128, help='Dims of generator hidden layer 1')
  parser.add_argument('--g_h2_dim', type=int, default=300, help='Dims of generator hidden layer 2')
  parser.add_argument('--d_h1_dim', type=int, default=128, help='Dims of discriminator hidden layer 1')
  parser.add_argument('--d_h2_dim', type=int, default=150, help='Dims of discriminator hidden layer 2')

  #opt details
  parser.add_argument('--log_period', type=int, default=50, help='anneal period')
  parser.add_argument('--batch_size', type=int, default=128, help='Example numbers every batch')
  parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
  parser.add_argument('--nb_epochs', type=int, default=100, help='Number of epoch')
  parser.add_argument('--max_grad_norm', type=float, default=10.0, help='Max norm of gradient')
  parser.add_argument('--gen_period', type=int, default=10, help='The period to run generator')

  return parser.parse_args()