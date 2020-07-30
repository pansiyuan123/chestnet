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

from tqdm import tqdm

parser = argparse.ArgumentParser(description='Chest-Xray Training')
parser.add_argument('--data_path', type=str,
                    default='./CVPR19/',
                    help='folder to load labels and images')
parser.add_argument('--save_path', type=str,
                    default='./chestx/logs2/',
                    help='folder to save output images and model checkpoints')
parser.add_argument('--image_size', type=int,
                    default=512,
                    help='image size (default: 512)')
parser.add_argument('--lr', '--learning_rate', type=float,
                    default=1e-4,
                    help='initial learning rate (default: 0.01)')
parser.add_argument('--lrp', type=float,
                    default=0.1,
                    help='learning rate for pre-trained layers (default: 0.1)')
parser.add_argument('--momentum', type=float,
                    default=0.9,
                    help='momentum (default: 0.9)')
parser.add_argument('--epochs', type=int,
                    default=20,
                    help='umber of total epochs to run')
parser.add_argument('--batch_size', type=int,
                    default=1,
                    help='batch size for training model')
parser.add_argument('--weight-decay', type=float,
                    default=1e-5,
                    help='weight decay (default: 18-e-5)')
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

# def main_chest():

transform_test = transforms.Compose([
    transforms.Resize(args.image_size),
    transforms.FiveCrop(448),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    # transforms.Normalize(args.normalize_mean, args.normalize_var),
    transforms.Lambda(lambda crops: torch.stack([
        transforms.Normalize(args.normalize_mean, args.normalize_var)(crop) for crop in crops])),
])

print('Preparing testing set...')
testset = ChestXrayDataSet(root=args.data_path, set="train", transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                          shuffle=False, num_workers=2)
print('Dataset loading is done!')

# Define Chest-Xray model
net = densenet121_chest(num_classes=args.num_classes, kmax = 3).to(device)

for epoch in range(6, 11):
    net.load_state_dict(torch.load('./chestx/logs5/model_{}.pth'.format(epoch)))
    print("Epoch is :{}".format(epoch))

    # data_loader = tqdm(testloader, desc='Test', mininterval=10)
    # print("Waiting for Test...")
    test_time = time.time()
    with torch.no_grad():
        gt, pred, ll = torch.FloatTensor(), torch.FloatTensor(), torch.FloatTensor()
        net.eval()
        for (inputs, labels) in testloader:
            inputs = inputs.view([args.batch_size*5, 3, 448, 448])

            gt = torch.cat((gt, labels), 0)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            outputs = outputs.view([args.batch_size, 5, args.num_classes]).max(dim=1)[0]
            pred = torch.cat((pred, outputs.cpu()), 0)
            loss = labels - outputs


        gt_npy = gt.cpu().numpy()
        pred_npy = pred.cpu().numpy()
        AUROCs = compute_AUCs(gt_npy, pred_npy)
        AUROC_avg = np.array(AUROCs).mean()
        # print('The average AUROC is {AUROC_avg:.4f}'.format(AUROC_avg=AUROC_avg))
        # for idx in range(args.num_classes):
        #     print('The AUROC of {} is {}'.format(args.classes[idx], AUROCs[idx]))
        # print('Testing cost time : {:.2f}'.format(time.time() - test_time))
        print('{AUROC_avg:.4f}'.format(AUROC_avg=AUROC_avg))
        for idx in range(args.num_classes):
            print('{}'.format(AUROCs[idx]))
        print('Testing cost time : {:.2f}'.format(time.time() - test_time))

