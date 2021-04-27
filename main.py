import argparse
import os
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import resnet
import utils
from pathlib import Path


def get_args_parser():
    parser = argparse.ArgumentParser('Propert ResNets for CIFAR10 in pytorch', add_help=False)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=50, type=int,
                        metavar='N', help='print frequency (default: 50)')
    parser.add_argument('--resume', default=None, type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--save-every', dest='save_every',
                        help='Saves checkpoints at every specified number of epochs',
                        type=int, default=10)
    parser.add_argument('--output_dir', default='logs', help='path where to save logs')
    parser.add_argument('--seed', default=11, type=int)
    parser.add_argument('--device', default='cuda', type=str,
                        help='device to use for training / testing')
    parser.add_argument('--GPU_ids', type=str, default='0', help='Ids of GPUs')
    return parser


def prepare_training(args):
    model = torch.nn.DataParallel(resnet.resnet32())
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[100, 150], last_epoch=args.start_epoch - 1)

    log('resnet32: #params={}'.format(utils.compute_num_params(model, text=True)))
    log(model)
    log(criterion)
    log('Building dataset...')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.resume:
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

    return model, criterion, optimizer, lr_scheduler, train_loader, val_loader


def train(train_loader, model, criterion, optimizer, epoch, device):
    """
        Run one train epoch
    """
    t = {'losses': utils.AverageMeter(),
         'top1': utils.AverageMeter()}
    for i in range(10):
        t[i] = utils.AverageMeter()
    log('Epoch: [{}]'.format(epoch))

    # switch to train mode
    model.train()
    criterion.train()

    for i, (input, target) in enumerate(train_loader):
        target = target.to(device)
        input_var = input.to(device)
        target_var = target

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1, num_map, correct_map = utils.accuracy(output.data, target)
        t['losses'].update(loss.item(), input.size(0))
        t['top1'].update(sum=prec1[0].item(), n=input.size(0))
        for j in range(10):
            t[j].update(sum=correct_map[j], n=num_map[j])

        if i % 50 == 0:
            print('step {}/{}:'.format(i, len(train_loader)))
            for k, v in t.items():
                print('{}:{:.4f}'.format(k, v.avg))
    return t


@torch.no_grad()
def validate(val_loader, model, criterion, device, epoch):
    """
    Run evaluation
    """
    model.eval()
    criterion.eval()
    t = {'losses': utils.AverageMeter(),
         'top1': utils.AverageMeter()}
    for i in range(10):
        t[i] = utils.AverageMeter()
    total_steps = len(val_loader)
    iterats = iter(val_loader)
    for step in range(total_steps):
        input, target = next(iterats)
        target = target.to(device)
        input_var = input.to(device)
        target_var = target
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1, num_map, correct_map = utils.accuracy(output.data, target)
        t['losses'].update(loss.item(), input.size(0))
        t['top1'].update(sum=prec1[0].item(), n=input.size(0))
        for j in range(10):
            t[j].update(sum=correct_map[j], n=num_map[j])
    return t


def main(args):
    global log, writer
    log, writer = utils.set_save_path(args.output_dir)
    log(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, optimizer, lr_scheduler, train_loader, val_loader = prepare_training(args)
    model.to(device)
    criterion.to(device)
    cudnn.benchmark = True

    if args.evaluate:
        validate(val_loader, model, criterion, device, 0)
        return

    # training
    output_dir = Path(args.output_dir)
    log("Start training")
    timer = utils.Timer()
    for epoch in range(args.start_epoch, args.epochs + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, args.epochs)]
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        train_stats = train(train_loader, model, criterion, optimizer, epoch, device)
        lr_scheduler.step()

        log_info.append('train:')
        log_info = log_info + ['{}={:.4f}'.format(k, v.avg) for k, v in train_stats.items()]

        if args.output_dir:
            checkpoint_path = output_dir / 'checkpoint.pth'
            if epoch > 0 and epoch % args.save_every == 0:
                sv_file = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'args': args
                }
                torch.save(sv_file, checkpoint_path)

        test_stats = validate(val_loader, model, criterion, device, epoch)
        log_info.append('eval:')
        log_info = log_info + ['{}={:.4f}'.format(k, v.avg) for k, v in test_stats.items()]

        writer.add_scalars('loss', {'train': train_stats['losses'].avg,
                                    'eval': test_stats['losses'].avg}, epoch)
        writer.add_scalars('Acc', {'train': train_stats['top1'].avg,
                                    'eval': test_stats['top1'].avg}, epoch)
        writer.add_figure('eval', utils.generate_fig(test_stats), epoch)
        writer.add_figure('train', utils.generate_fig(train_stats), epoch)

        t = timer.t()
        prog = (epoch - args.start_epoch + 1) / (args.epochs - args.start_epoch + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))
        writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Cifar-10 training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.GPU_ids)
    main(args)
