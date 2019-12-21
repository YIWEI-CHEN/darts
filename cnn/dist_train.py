import argparse
import glob
import logging
import logging.handlers
import multiprocessing
import sys
import threading
import time

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

import utils
import genotypes
from model import NetworkCIFAR as Network

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=str, default="0,1", help='gpu device id')
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
parser.add_argument('--exec_script', type=str, default='scripts/search.sh', help='script to run exp')
parser.add_argument('--dist-url', default='tcp://datalab.cse.tamu.edu:50017', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('--train_portion', type=float, default=0.9, help='portion of training data')
CIFAR_CLASSES = 10
best_acc1 = 0


def main():
  try:
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'), exec_script=args.exec_script)

    # Logging configuration
    utils.setup_logger(args)
    root = logging.getLogger()

    if not torch.cuda.is_available():
      root.info('no gpu device available')
      sys.exit(1)

    # Fix seed
    utils.fix_seed(args.seed)

    # Log thread to receive log from child processes
    ctx = multiprocessing.get_context('spawn')
    log_queue = ctx.Queue()
    log_thread = threading.Thread(target=_handle_log, name='log_thread', args=(log_queue,))
    log_thread.start()

    root.info('gpu device = %s' % args.gpu)
    root.info("args = %s", args)

    ngpus_per_node = torch.cuda.device_count()
    # Since we have ngpus_per_node processes per node, the total world_size
    # needs to be adjusted accordingly
    args.world_size = ngpus_per_node * args.world_size
    # Use torch.multiprocessing.spawn to launch distributed processes: the
    # main_worker process function
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, log_queue))
  finally:
    # close the thread
    log_queue.put(None)
    log_thread.join(1.0)


def main_worker(gpu, ngpus_per_node, args, log_queue):
  qh = logging.handlers.QueueHandler(log_queue)
  root = logging.getLogger()
  root.setLevel(logging.DEBUG)
  root.addHandler(qh)

  args.gpu = gpu
  global best_acc1

  try:
    # For multiprocessing distributed training, rank needs to be the
    # global rank among all the processes
    args.rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    # create model
    genotype = eval("genotypes.%s" % args.arch)
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
    root.info("param size = %fMB", utils.count_parameters_in_MB(model))

    # For multiprocessing distributed, DistributedDataParallel constructor
    # should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices.
    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)

    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    args.batch_size = int(args.batch_size / ngpus_per_node)
    args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay
    )

    # Data loading code
    train_queue, train_sampler, valid_queue = utils.get_train_validation_loader(args)
    test_queue = utils.get_test_loader(args)

    # learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

    for epoch in range(args.epochs):
      train_sampler.set_epoch(epoch)
      scheduler.step()
      root.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
      model.module.drop_path_prob = args.drop_path_prob * epoch / args.epochs

      # train for one epoch
      train_acc, train_obj = utils.train(train_queue, model, criterion, optimizer, args)
      root.info('train_acc %f, train_loss %f', train_acc, train_obj)

      # evaluate on validation set
      valid_acc, valid_obj = utils.infer(valid_queue, model, criterion, args)
      root.info('valid_acc %f, valid_obj %f', valid_obj, valid_obj)

      # evaluate on test set
      test_acc, test_obj = utils.infer(test_queue, model, criterion, args)
      root.info('test_acc %f, test_obj %f', test_acc, test_obj)

      # remember best acc@1 and save checkpoint
      is_best = valid_acc > best_acc1
      best_acc1 = max(valid_acc, best_acc1)

      if args.rank % ngpus_per_node == 0:
        utils.save_checkpoint({
          'epoch': epoch + 1,
          'arch': args.arch,
          'state_dict': model.state_dict(),
          'best_acc1': best_acc1,
          'optimizer': optimizer.state_dict(),
        }, is_best, args.save)
        if is_best:
          root.info('best test_acc: {} at Epoch {} when valid_acc is {}'.format(
            test_acc, epoch, best_acc1
          ))
  finally:
    dist.destroy_process_group()


def _handle_log(log_queue):
  while True:
    record = log_queue.get()
    if record is None:
      break
    logger = logging.getLogger(record.name)
    logger.handle(record)


if __name__ == '__main__':
  main() 

