import logging
import multiprocessing
import sys
import threading

import random

import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.backends import cudnn
import torchvision.datasets as dset

from sampler import DistributedSubsetSampler, SubsetSampler

class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def create_exp_dir(path, scripts_to_save=None, exec_script='scripts/exec.sh'):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)
    dst_file = os.path.join(path, os.path.basename(exec_script))
    shutil.copyfile(exec_script, dst_file)


def fix_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  cudnn.benchmark = False
  torch.manual_seed(seed)
  cudnn.enabled = True
  cudnn.deterministic = True
  torch.cuda.manual_seed(seed)


def setup_logger(args):
  # log_format = '%(asctime)s %(processName)-15s [%(filename)s:%(lineno)d] %(message)s'
  log_format = '%(asctime)s %(processName)-15s %(message)s'
  logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                      format=log_format, datefmt='%m/%d %I:%M:%S %p')
  fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
  fh.setFormatter(logging.Formatter(log_format))
  logging.getLogger().addHandler(fh)


def train(train_queue, model, criterion, optimizer, args):
  root = logging.getLogger()
  objs = AvgrageMeter()
  top1 = AvgrageMeter()
  top5 = AvgrageMeter()

  # switch to train mode
  model.train()

  for step, (input, target) in enumerate(train_queue):
    input = input.cuda(args.gpu, non_blocking=True)
    target = target.cuda(args.gpu, non_blocking=True)

    # compute output
    logits, logits_aux = model(input)
    loss = criterion(logits, target)

    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight * loss_aux

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    # measure accuracy and record loss
    prec1, prec5 = accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)

    if step % args.report_freq == 0:
      root.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def infer(queue, model, criterion, args, search=False):
  root = logging.getLogger()
  objs = AvgrageMeter()
  top1 = AvgrageMeter()
  top5 = AvgrageMeter()

  # switch to evaluate mode
  model.eval()

  with torch.no_grad():
    for step, (input, target) in enumerate(queue):
      input = input.cuda(args.gpu, non_blocking=True)
      target = target.cuda(args.gpu, non_blocking=True)

      # compute output
      if search:
        logits = model(input)
      else:
        logits, _ = model(input)
      loss = criterion(logits, target)

      # measure accuracy and record loss
      prec1, prec5 = accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      objs.update(loss.data.item(), n)
      top1.update(prec1.data.item(), n)
      top5.update(prec5.data.item(), n)

      if step % args.report_freq == 0:
        root.info('%s %03d %e %f %f', queue.name, step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def search_train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, args):
    root = logging.getLogger()
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):
        # switch to train mode
        model.train()
        n = input.size(0)

        input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # get a random minibatch from the search queue with replacement
        input_search, target_search = next(iter(valid_queue))
        input_search = input_search.cuda(args.gpu, non_blocking=True)
        target_search = target_search.cuda(args.gpu, non_blocking=True)

        # search architecture
        architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

        # compute output
        logits = model(input)
        loss = criterion(logits, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            root.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
        break

    return top1.avg, objs.avg


def get_train_validation_loader(args, distributed=True):
    train_transform, valid_transform = _data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    # train[0:split] as training data
    if distributed:
        train_sampler = DistributedSubsetSampler(train_data)
        train_sampler.set_split(split)
    else:
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    # train[split:] as validation data
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
    # valid_sampler = SubsetSampler(indices[split:])
    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, sampler=valid_sampler)
    valid_queue.name = 'valid'

    return train_queue, train_sampler, valid_queue


def get_test_loader(args):
    _, test_transform = _data_transforms_cifar10(args)
    test_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=test_transform)
    test_queue = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    test_queue.name = 'test'
    return test_queue


def run_log_thread():
    log_queue = multiprocessing.get_context('spawn').Queue()

    def _handle_log():
        while True:
            record = log_queue.get()
            if record is None:
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)

    log_thread = threading.Thread(target=_handle_log, name='log_thread')
    log_thread.start()

    return log_thread, log_queue
