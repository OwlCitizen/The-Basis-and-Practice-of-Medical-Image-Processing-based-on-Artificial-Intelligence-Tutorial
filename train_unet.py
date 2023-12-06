#train_unet.py

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from dataset import ISBI_Dataset
from torch.utils.data import DataLoader
from model import UNet
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import time
from utils import dice_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from tensorboardX import SummaryWriter

def weighted_loss(out, label, cls_criterion):
    out = out.transpose(1,2).transpose(2,3).flatten(1,2).flatten(0,1).float()
    label = label.flatten(1,2).flatten(0,1).long()
    
    poi = (label == 1).nonzero()
    nei = (label == 0).nonzero()
    pw = len(nei)*1.0/len(label)
    nw = len(poi)*1.0/len(label)
    assert pw+nw == 1
    po_loss = pw*cls_criterion(out[poi].squeeze(), label[poi].squeeze())
    ne_loss = nw*cls_criterion(out[nei].squeeze(), label[nei].squeeze())
    loss = po_loss + ne_loss
    return loss

def dice_loss(out, label):
    pred_mask = F.softmax(out, dim = 1).transpose(1,2).transpose(2,3).clone().detach_().cpu()[:,:,:,1]
    gt_mask = label.transpose(1,2).transpose(2,3).squeeze()
    return 1-dice_score(pred_mask, gt_mask)

def accuracy(out, label):
    #output: [b, c, h, w]
    #target: [b, 1, h, w]
    with torch.no_grad():
        pred = F.softmax(out, dim = 1).transpose(1,2).transpose(2,3).flatten(1,2).clone().detach_().cpu()
        gt = label.transpose(1,2).transpose(2,3).flatten(1,2)
        acc = accuracy_score(gt.long().flatten(0,1), pred.argmax(dim = -1).flatten(0,1))
        
        pred_mask = F.softmax(out, dim = 1).transpose(1,2).transpose(2,3).clone().detach_().cpu()[:,:,:,1]
        gt_mask = label.transpose(1,2).transpose(2,3).squeeze()
        dice = dice_score(pred_mask, gt_mask)
        
        return acc, dice.clone().detach_().cpu().item() 

def save_checkpoint(state, is_best, filename='net_params.pt'):
    print('saving parameters ...')
    torch.save(state, './net_params/'+filename)
    if is_best:
        torch.save(state, './net_params/'+filename.split('.')[-2]+'_best.'+filename.split('.')[-1])
    print('Net params Saved!')

def train(train_loader, model, criterion, optimizer, epoch, writer, criterion_e = None):
    """
        训练代码
        参数：
            train_loader - 训练集的 DataLoader
            model - 模型
            criterion - 损失函数
            optimizer - 优化器
            epoch - 进行第几个 epoch
            writer - 用于写 tensorboardX 
            criterion_e - 额外损失函数
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (batch, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        x = F.pad(batch.float(), [4, 4, 4, 4]).cuda()
        y = target.long().squeeze().cuda()

        # compute output
        losses_step = []
        output = model(x)
        loss = weighted_loss(output, y, criterion)
        losses_step.append(loss)
        if criterion_e is not None:
            extra_loss = criterion_e(output, target)
            loss = loss*0.1 + extra_loss
            losses_step.append(extra_loss)
        print('\tEpoch ',epoch, 'Step',i,'Loss:',losses_step)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target)
        losses.update(loss.item(), batch.size(0))
        top1.update(prec1, batch.size(0))
        top5.update(prec5, batch.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Dice {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
    writer.add_scalar('loss/train_loss', losses.val, global_step=epoch)

def validate(val_loader, model, criterion, epoch, writer, phase="VAL"):
    """
        验证代码
        参数：
            val_loader - 验证集的 DataLoader
            model - 模型
            criterion - 损失函数
            epoch - 进行第几个 epoch
            writer - 用于写 tensorboardX 
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (batch, target) in enumerate(val_loader):
            x = F.pad(batch.float(), [4, 4, 4, 4]).cuda()
            y = target.long().squeeze().cuda()
            # compute output
            output = model(x)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target)
            losses.update(loss.item(), batch.size(0))
            top1.update(prec1, batch.size(0))
            top5.update(prec5, batch.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                print('Test-{0}: [{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Dice {top5.val:.3f} ({top5.avg:.3f})'.format(
                              phase, i, len(val_loader),
                              batch_time=batch_time,
                              loss=losses,
                              top1=top1, top5=top5))

        print(' * {} Acc {top1.avg:.3f} Dice {top5.avg:.3f}'
              .format(phase, top1=top1, top5=top5))
    writer.add_scalar('loss/valid_loss', losses.val, global_step=epoch)
    return top1.avg, top5.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == "__main__":
    # -------------------------------------------- step 1/4 : 加载数据 ---------------------------
    train_dir_list = 'train.txt'
    valid_dir_list = 'val.txt'
    batch_size = 4
    epochs = 80
    num_classes = 2
    train_data = ISBI_Dataset(data_path = './data/train/image/')
    valid_data = ISBI_Dataset(data_path = './data/test')
    train_loader = DataLoader(
        dataset=train_data, 
        batch_size=batch_size, 
        shuffle=True)
    valid_loader = DataLoader(
        dataset=valid_data, 
        batch_size=batch_size, 
        shuffle=False)
    train_data_size = len(train_data)
    print('训练集数量：%d' % train_data_size)
    valid_data_size = len(valid_data)
    print('验证集数量：%d' % valid_data_size)
    # ------------------------------------ step 2/4 : 定义网络 ------------------------------------
    model = UNet(n_channels=1, n_classes=2, bilinear = False).cuda()
    #fc_inputs = model.fc.in_features
    #model.fc = nn.Linear(fc_inputs, num_classes)
    #model = model.cuda()
    # ------------------------------------ step 3/4 : 定义损失函数和优化器等 -------------------------
    lr_init = 0.0001
    lr_stepsize = 20
    weight_decay = 0.001
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr_init, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_stepsize, gamma=0.1)
    
    writer = SummaryWriter('./tensorboard/train_unet/')
    # ------------------------------------ step 4/4 : 训练 -----------------------------------------
    best_prec1 = 0
    for epoch in range(epochs):
        scheduler.step()
        train(train_loader, model, criterion, optimizer, epoch, writer, criterion_e = dice_loss)
        # 在验证集上测试效果
        valid_prec1, valid_prec5 = validate(valid_loader, model, dice_loss, epoch, writer, phase="VAL")
        is_best = valid_prec5 > best_prec1
        best_prec1 = max(valid_prec5, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'UNet',
            'state_dict': model.state_dict(),
            'best_dice': best_prec1,
            'optimizer' : optimizer.state_dict(),
            }, is_best)
    writer.close()
    print('Finish!')
