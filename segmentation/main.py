import argparse
import os
import tensorflow as tf

tf.set_random_seed(19)
from model import hypergae


parser = argparse.ArgumentParser(description='')

parser.add_argument('--dataset_dir', dest='dataset_dir', default='MulModal', help='path of the dataset')

parser.add_argument('--epoch', dest='epoch', type=int, default=1501, help='# of epoch')
parser.add_argument('--warmup_epoch', dest='warmup_epoch', type=int, default=10, help='# of warmup_epoch')

parser.add_argument('--batch_size', dest='batch_size', type=int, default=2, help='# images in batch')
parser.add_argument('--max_update_num', dest='max_update_num', type=int, default=2000, help='# updates at each epoch')

parser.add_argument('--n_domains', dest='n_domains', type=int, default=4, help='# domain numbers in multi-modal synthesis')
parser.add_argument('--m_labels', dest='m_labels', type=int, default=3, help='# label numbers in segmentation')

parser.add_argument('--crop_size', dest='crop_size', type=int, default=72, help='crop volume to this size')

parser.add_argument('--ngf', dest='ngf', type=int, default=24, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=24, help='# of dis filters in first conv layer')

parser.add_argument('--nsf', dest='nsf', type=int, default=16, help='# of seg filters in first conv layer')

parser.add_argument('--image_dim', dest='image_dim', type=int, default=1, help='# of image channels')

parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--phase', dest='phase', default='train', help='train, valid, test')

parser.add_argument('--save_freq', dest='save_freq', type=int, default=10, help='save a model every save_freq epochs')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=100, help='print the debug information every print_freq iterations')
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=True, help='if continue training, load the latest model: 1: true, 0: false')

parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')

parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=0.8, help='weight on loss_recon term in objective')
parser.add_argument('--L3_lambda', dest='L3_lambda', type=float, default=0.001, help='weight on loss_acf term in objective')

args = parser.parse_args()


def main(_):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

    gpu_id = 1
    # gpu = '/gpu:' + str(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    
    with tf.Session(config=tfconfig) as sess:
        model = hypergae(sess, args)
        
        if args.phase == 'train':
            model.train(args) 
        elif args.phase == 'valid':
            model.valid(args)
        else: 
            model.test(args)


if __name__ == '__main__':
    tf.app.run()
