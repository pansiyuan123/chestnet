import os
import time
import argparse
import numpy as np

listdir = os.getcwd()
os.chdir(os.path.dirname(listdir))

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from tensorboardX import SummaryWriter
from sklearn.metrics import roc_auc_score
from chestx.models import densenet121_chest
from chestx.nih import ChestXrayDataSet
from chestx.loss import FocalLoss, MultiLabelBCELoss

parser = argparse.ArgumentParser(description='Chest-Xray Training')
parser.add_argument('--data_path', type=str,
                    default='./CVPR19/',
                    help='folder to load labels and images')
parser.add_argument('--save_path', type=str,
                    default='./chestx/logs_base/',
                    help='folder to save output images and model checkpoints')
parser.add_argument('--image_size', type=int,
                    default=512,
                    help='image size (default: 512)')
parser.add_argument('--lr', '--learning_rate', type=float,
                    default=1e-3,
                    help='initial learning rate (default: 0.01)')
parser.add_argument('--lrp', type=float,
                    default=0.1,
                    help='learning rate for pre-trained layers (default: 0.1)')
parser.add_argument('--momentum', type=float,
                    default=0.9,
                    help='momentum (default: 0.9)')
parser.add_argument('--epochs', type=int,
                    default=60,
                    help='umber of total epochs to run')
parser.add_argument('--batch_size', type=int,
                    default=15,
                    help='batch size for training model')
parser.add_argument('--test_batch_size', type=int,
                    default=2,
                    help='batch size for testing model')
parser.add_argument('--weight-decay', type=float,
                    default=1e-4,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--num_classes', type=int,
                    default=14,
                    help='The numbers of classes')
parser.add_argument('--normalize_mean', type=tuple,
                    default=(0.5, 0.5, 0.5),
                    help='mean value for image normalization')
parser.add_argument('--normalize_var', type=tuple,
                    default=(0.5, 0.5, 0.5),
                    help='variance value for image normalization')
parser.add_argument('--classes', type=tuple,
                    default=('Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'),
                    help='labels of Chest-Xray')
args = parser.parse_args()

# Define to use GPU
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def compute_AUCs(gt_np, pred_np):
    AUROCs = []
    for i in range(args.num_classes):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs

def main_chest():
    print('chong xian')
    # Preparing for Dataset
    transform_train = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.RandomCrop(448),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(args.normalize_mean, args.normalize_var)
    ])

    transform_test = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.FiveCrop(448),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        # transforms.Normalize(args.normalize_mean, args.normalize_var),
        transforms.Lambda(lambda crops: torch.stack([
            transforms.Normalize(args.normalize_mean, args.normalize_var)(crop) for crop in crops])),
    ])

    print('Preparing training set...')
    trainset = ChestXrayDataSet(root=args.data_path, set="train", transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=2)

    print('Preparing testing set...')
    testset = ChestXrayDataSet(root=args.data_path, set="test", transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size,
                                              shuffle=False, num_workers=2)
    print('Dataset loading is done!')

    # Define Chest-Xray model
    net = densenet121_chest(num_classes=args.num_classes, kmax=1, kmin=1, alpha=0.7, num_maps=12).to(device)

    # Define Loss function and Optimizer
    criterion = MultiLabelBCELoss()
    criterion = criterion.to(device)
    optimizer = torch.optim.SGD(net.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[9, 12, 15], gamma=0.1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    state = {'train_time':time.time(), 'test_time':time.time(), 'step':0,
                'learning_rate':scheduler.get_lr()[0], 'sum_loss':0.,}
    writer = SummaryWriter(args.save_path)
    # net.load_state_dict(torch.load('./chestx/logs2/model_4.pth'))

    # Training
    print("Staring Training ...")
    for epoch in range(1, args.epochs + 1):
        if epoch >= 5:
            scheduler.step()
        for i, (inputs, labels) in enumerate(trainloader, 0):
            net.train()
            length = len(trainloader)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            state['sum_loss'] += loss.data.item()
            state['step'] += 1
            if (i + 1) % 10 == 0:
                writer.add_scalar('Loss', loss.data.item(), (i + 1 + epoch * length))
            if (i + 1) % 1000 == 0:
                state['learning_rate'] = np.log10(1 / scheduler.get_lr()[0]).astype(int)
                print('[epoch : {}, iter : {}, lr : 1e-{}] Loss: {:.3f} Cost Time : {:.2f}'.format(
                      epoch, (i + 1 + (epoch - 1) * length), state['learning_rate'],
                        state['sum_loss'] / state['step'], time.time() - state['train_time']))
                state['sum_loss'], state['step'], state['train_time'] = 0., 0, time.time()

        print("Waiting for Test...")
        state['test_time'] = time.time()
        with torch.no_grad():
            gt, pred = torch.FloatTensor(), torch.FloatTensor()
            for (inputs, labels) in testloader:
                inputs = inputs.view([args.test_batch_size * 5, 3, 448, 448])

                net.eval()
                gt = torch.cat((gt, labels), 0)
                inputs = inputs.to(device)
                outputs = net(inputs)
                outputs = outputs.view([args.test_batch_size, 5, args.num_classes]).max(dim=1)[0]
                pred = torch.cat((pred, outputs.cpu()), 0)

            gt_npy = gt.cpu().numpy()
            pred_npy = pred.cpu().numpy()
            AUROCs = compute_AUCs(gt_npy, pred_npy)
            AUROC_avg = np.array(AUROCs).mean()
            print('The average AUROC is {AUROC_avg:.4f}'.format(AUROC_avg=AUROC_avg))
            for idx in range(args.num_classes):
                print('The AUROC of {} is {}'.format(args.classes[idx], AUROCs[idx]))
            print('Testing cost time : {:.2f}'.format(time.time() - state['test_time']))
        state['train_time'] = state['test_time'] = time.time()

        writer.add_scalar('AUROC', AUROC_avg, epoch)

        print("Saving model...")
        torch.save(net.state_dict(), '{}/model_{}.pth'.format(args.save_path, epoch))


if __name__ == '__main__':
    main_chest()