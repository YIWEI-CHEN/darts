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
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable

from resnet import ResNet18, ResNet50, ResNet34
from model import NetworkCIFAR as Network
from load_corrupted_data import CIFAR10, CIFAR100

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
# parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'],
#                     help='Choose between CIFAR-10, CIFAR-100.')
# parser.add_argument('--gold_fraction', '-gf', type=float, default=1, help='What fraction of the data should be trusted?')
parser.add_argument('--corruption_prob', '-cprob', type=float, default=0.7, help='The label corruption probability.')
parser.add_argument('--corruption_type', '-ctype', type=str, default='unif',
                    help='Type of corruption ("unif", "flip", hierarchical).')
parser.add_argument('--time_limit', type=int, default=12*60*60, help='Time limit for search')
parser.add_argument('--loss_func', type=str, default='cce', choices=['cce', 'rll', 'forward_gold'],
                    help='Choose between Categorical Cross Entropy (CCE), Robust Log Loss (RLL).')
parser.add_argument('--alpha', type=float, default=0.1, help='alpha for RLL')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--random_weight', action='store_true', default=False, help='use random weights')
parser.add_argument('--clean_train', action='store_true', default=False, help='use clean train')
args = parser.parse_args()

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

# if args.dataset == 'cifar10':
#   CIFAR_CLASSES = 10
# elif args.dataset == 'cifar100':
#   CIFAR_CLASSES = 100
# else:
CIFAR_CLASSES = 10

def weights_init(m):
  if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    torch.nn.init.xavier_uniform(m.weight)
    # if len(m.weight.data.shape) > 1:
    #   logging.info('{}, {}, {}'.format(m, m.weight.data.shape, m.weight.data[0][0]))
    # else:
    #   logging.info('{}, {}'.format(m, m.weight.data.shape, m.weight.data[0]))

def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  if args.arch == 'resnet':
    model = ResNet18(CIFAR_CLASSES).cuda()
    args.auxiliary = False
  elif args.arch == 'resnet50':
    model = ResNet50(CIFAR_CLASSES).cuda()
    args.auxiliary = False
  elif args.arch == 'resnet34':
    model = ResNet34(CIFAR_CLASSES).cuda()
    args.auxiliary = False
  else:
    genotype = eval("genotypes.%s" % args.arch)
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
    model = model.cuda()

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay
      )

  train_transform, test_transform = utils._data_transforms_cifar10(args)

  train_data = CIFAR10(
    root=args.data, train=True, gold=False, gold_fraction=0.0,
    corruption_prob=args.corruption_prob, corruption_type=args.corruption_type,
    transform=train_transform, download=True, seed=args.seed)
  gold_train_data = CIFAR10(
    root=args.data, train=True, gold=True, gold_fraction=1.0,
    corruption_prob=args.corruption_prob, corruption_type=args.corruption_type,
    transform=train_transform, download=True, seed=args.seed)

  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))

  clean_train_queue = torch.utils.data.DataLoader(
    gold_train_data, batch_size=args.batch_size,
    sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
    pin_memory=True, num_workers=2)

  noisy_train_queue = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size,
    sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
    pin_memory=True, num_workers=2)

  clean_train_list = []
  for input, target in clean_train_queue:
    clean_train_list.append((input, target))

  noisy_train_list = []
  for input, target in noisy_train_queue:
    noisy_train_list.append((input, target))

  clean_valid_queue = torch.utils.data.DataLoader(
    gold_train_data, batch_size=args.batch_size,
    sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:]),
    pin_memory=True, num_workers=2)

  noisy_valid_queue = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size,
    sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:]),
    pin_memory=True, num_workers=2)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

  if args.loss_func == 'cce':
    criterion = nn.CrossEntropyLoss().cuda()
  elif args.loss_func == 'rll':
    criterion = utils.RobustLogLoss(alpha=args.alpha).cuda()
  elif args.loss_func == 'forward_gold':
    corruption_matrix = train_data.corruption_matrix
    criterion = utils.ForwardGoldLoss(corruption_matrix=corruption_matrix)
  else:
    assert False, "Invalid loss function '{}' given. Must be in {'cce', 'rll'}".format(args.loss_func)

  for epoch in range(args.epochs):
    if args.random_weight:
      logging.info('Randomly assign weights')
      model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
      torch.manual_seed(epoch)
      clean_obj, noisy_obj = infer_random_weight(clean_train_list, noisy_train_list, model, criterion)
      logging.info('clean loss %f, noisy loss %f', clean_obj, noisy_obj)
    else:
      scheduler.step()
      logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
      model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

      # train_queue = clean_train_queue if args.clean_train else noisy_train_queue
      train_acc, train_obj, another_obj = train(clean_train_list, noisy_train_list, model, criterion, optimizer)
      if args.clean_train:
        logging.info('train_acc %f, clean_loss %f, noisy_loss %f', train_acc, train_obj, another_obj)
      else:
        logging.info('train_acc %f, clean_loss %f, noisy_loss %f', train_acc, another_obj, train_obj)

      # clean_valid_acc, valid_obj = infer_valid(clean_valid_queue, model, criterion)
      # logging.info('clean_valid_acc %f', clean_valid_acc)

      # noisy_valid_acc, valid_obj = infer_valid(noisy_valid_queue, model, criterion)
      # logging.info('noisy_valid_acc %f', noisy_valid_acc)

      # utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(clean_train_list, noisy_train_list, model, criterion, optimizer):
  objs = utils.AvgrageMeter()
  another_objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.train()

  if args.clean_train:
    train_list = clean_train_list
    another_list = noisy_train_list
  else:
    train_list = noisy_train_list
    another_list = clean_train_list

  for step, (input, target) in enumerate(train_list):
    another_input, another_target = another_list[step]

    input = Variable(input).cuda()
    target = Variable(target).cuda(async=True)

    optimizer.zero_grad()
    logits, logits_aux = model(input)
    loss = criterion(logits, target)
    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight*loss_aux

    another_input = Variable(another_input).cuda()
    another_target = Variable(another_target).cuda(async=True)

    another_logits, logits_aux = model(another_input)
    another_loss = criterion(another_logits, another_target)
    if args.auxiliary:
      loss_aux = criterion(logits_aux, another_target)
      another_loss += args.auxiliary_weight*loss_aux

    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data[0], n)
    top1.update(prec1.data[0], n)
    top5.update(prec5.data[0], n)

    n = another_input.size(0)
    another_objs.update(another_loss.data[0], n)

    if step % args.report_freq == 0:
      if args.clean_train:
        logging.info('train %03d clean %e noisy %e %f %f', step, objs.avg, another_objs.avg, top1.avg, top5.avg)
      else:
        logging.info('train %03d clean %e noisy %e %f %f', step, another_objs.avg, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg, another_objs.avg

def infer_random_weight(clean_train_list, noisy_train_list, model, criterion):
  clean_objs = utils.AvgrageMeter()
  noisy_objs = utils.AvgrageMeter()
  model.train()
  model.apply(weights_init)
  nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
  # model.eval()

  total_batch = len(clean_train_list)
  for step in range(total_batch):
    for train_list, objs in [
      (clean_train_list, clean_objs),
      (noisy_train_list, noisy_objs),
    ]:
      input_, target_ = train_list[step]
      input_ = Variable(input_).cuda()
      target_ = Variable(target_).cuda(async=True)

      logits, logits_aux = model(input_)
      loss = criterion(logits, target_)

      if args.auxiliary:
        loss_aux = criterion(logits_aux, target_)
        loss += args.auxiliary_weight * loss_aux

      n = input_.size(0)
      objs.update(loss.data[0], n)

    if step % args.report_freq == 0:
        logging.info('step %03d clean loss %e, noisy loss %e', step, clean_objs.avg, noisy_objs.avg)

  return clean_objs.avg, noisy_objs.avg

def infer_valid(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda(async=True)

    logits, _ = model(input)
    loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data[0], n)
    top1.update(prec1.data[0], n)
    top5.update(prec5.data[0], n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main() 

