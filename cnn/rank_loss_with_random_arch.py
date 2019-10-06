import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random

from torch.autograd import Variable
from model_search import Network
from architect import Architect
from genotypes import PRIMITIVES

from load_corrupted_data import CIFAR10, CIFAR100

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
# parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
# parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
# parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
# parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
# parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
# parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
# parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'],
                    help='Choose between CIFAR-10, CIFAR-100.')
# parser.add_argument('--gold_fraction', '-gf', type=float, default=1, help='What fraction of the data should be trusted?')
parser.add_argument('--corruption_prob', '-cprob', type=float, default=0.7, help='The label corruption probability.')
parser.add_argument('--corruption_type', '-ctype', type=str, default='unif',
                    help='Type of corruption ("unif", "flip", hierarchical).')
parser.add_argument('--time_limit', type=int, default=12*60*60, help='Time limit for search')
parser.add_argument('--loss_func', type=str, default='cce', choices=['cce', 'rll'],
                    help='Choose between Categorical Cross Entropy (CCE), Robust Log Loss (RLL).')
# parser.add_argument('--clean_valid', action='store_true', default=False, help='use clean validation')
parser.add_argument('--alpha', type=float, default=0.1, help='alpha for RLL')
args = parser.parse_args()

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

if args.dataset == 'cifar10':
  CIFAR_CLASSES = 10
elif args.dataset == 'cifar100':
  CIFAR_CLASSES = 100
else:
  CIFAR_CLASSES = 10

def weights_init(m):
  if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    torch.nn.init.xavier_uniform(m.weight)

def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = False
  torch.manual_seed(args.seed)
  cudnn.enabled = True
  cudnn.deterministic = True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  if args.loss_func == 'cce':
    criterion = nn.CrossEntropyLoss().cuda()
  elif args.loss_func == 'rll':
    criterion = utils.RobustLogLoss(alpha=args.alpha).cuda()
  else:
    assert False, "Invalid loss function '{}' given. Must be in {'cce', 'rll'}".format(args.loss_func)

  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
  model = model.cuda()
  model.train()
  model.apply(weights_init)
  nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  train_transform, valid_transform = utils._data_transforms_cifar10(args)

  # Load dataset
  if args.dataset == 'cifar10':
    train_data = CIFAR10(
      root=args.data, train=True, gold=False, gold_fraction=0.0,
      corruption_prob=args.corruption_prob, corruption_type=args.corruption_type,
      transform=train_transform, download=True, seed=args.seed)
    gold_train_data = CIFAR10(
      root=args.data, train=True, gold=True, gold_fraction=1.0,
      corruption_prob=args.corruption_prob, corruption_type=args.corruption_type,
      transform=train_transform, download=True, seed=args.seed)
  elif args.dataset == 'cifar100':
    train_data = CIFAR100(
      root=args.data, train=True, gold=False, gold_fraction=0.0,
      corruption_prob=args.corruption_prob, corruption_type=args.corruption_type,
      transform=train_transform, download=True, seed=args.seed)
    gold_train_data = CIFAR100(
      root=args.data, train=True, gold=True, gold_fraction=1.0,
      corruption_prob=args.corruption_prob, corruption_type=args.corruption_type,
      transform=train_transform, download=True, seed=args.seed)
  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))

  clean_train_queue = torch.utils.data.DataLoader(
      gold_train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=0)
  noisy_train_queue = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size,
    sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
    pin_memory=True, num_workers=0)

  clean_valid_queue = torch.utils.data.DataLoader(
    gold_train_data, batch_size=args.batch_size,
    sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:]),
    pin_memory=True, num_workers=0)
  noisy_valid_queue = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size,
    sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:]),
    pin_memory=True, num_workers=0)

  for epoch in range(args.epochs):
    logging.info('Epoch %d, random architecture with fix weights', epoch)

    genotype = model.genotype()
    logging.info('genotype = %s', genotype)

    logging.info(F.softmax(model.alphas_normal, dim=-1))
    logging.info(F.softmax(model.alphas_reduce, dim=-1))

    # training
    clean_train_acc, clean_train_obj = infer(clean_train_queue, model, criterion, kind='clean_train')
    logging.info('clean_train_acc %f, clean_train_loss %f', clean_train_acc, clean_train_obj)

    noisy_train_acc, noisy_train_obj = infer(noisy_train_queue, model, criterion, kind='noisy_train')
    logging.info('noisy_train_acc %f, noisy_train_loss %f', noisy_train_acc, noisy_train_obj)

    # validation
    clean_valid_acc, clean_valid_obj = infer(clean_valid_queue, model, criterion, kind='clean_valid')
    logging.info('clean_valid_acc %f, clean_valid_loss %f', clean_valid_acc, clean_valid_obj)

    # validation
    noisy_valid_acc, noisy_valid_obj = infer(noisy_valid_queue, model, criterion, kind='noisy_valid')
    logging.info('noisy_valid_acc %f, noisy_valid_loss %f', noisy_valid_acc, noisy_valid_obj)

    utils.save(model, os.path.join(args.save, 'weights.pt'))

    # Randomly change the alphas
    k = sum(1 for i in range(model._steps) for n in range(2 + i))
    num_ops = len(PRIMITIVES)
    model.alphas_normal.data.copy_(torch.randn(k, num_ops))
    model.alphas_reduce.data.copy_(torch.randn(k, num_ops))

def infer(valid_queue, model, criterion, kind):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda(async=True)

    logits = model(input)
    loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data[0], n)
    top1.update(prec1.data[0], n)
    top5.update(prec5.data[0], n)

    if step % args.report_freq == 0:
      logging.info('%s %03d %e %f %f', kind, step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main() 

