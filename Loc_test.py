import os
import cv2
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

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
                    default='./chestx/logs19/',
                    help='folder to save output images and model checkpoints')
parser.add_argument('--image_size', type=int,
                    default=512,
                    help='image size (default: 512)')
parser.add_argument('--lr', '--learning_rate', type=float,
                    default=1e-2,
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
                    default=15,
                    help='batch size for training model')
parser.add_argument('--test_batch_size', type=int,
                    default=11,
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

# Define to use GPU0
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def compute_AUCs(gt_np, pred_np):
    AUROCs = []
    for i in range(args.num_classes):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs

def main_chest():
    print('222')
    # Preparing for Dataset
    transform_test = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(448),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(args.normalize_mean, args.normalize_var)
    ])

    print('Preparing testing set...')
    testset = ChestXrayDataSet(root=args.data_path, set="loc", transform=transform_test, isLoc=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size,
                                             shuffle=False, num_workers=2)

    # Define Chest-Xray model
    net = densenet121_chest(num_classes=args.num_classes, kmax=5).to(device)
    net.load_state_dict(torch.load('./chestx/logs5/model_10.pth'))

    with torch.no_grad():
        net.eval()
        for i, (img, inputs, labels, ill, loc) in enumerate(testloader, 0):
            inputs = inputs.to(device)

            outputs, hmask = net(inputs)
            outputs = outputs.argmax(1).cpu()

            hmask = hmask.cpu()
            hmask = hmask.numpy()
            for i in range(args.test_batch_size):
                if not outputs[i].equal(ill[i]):
                    continue
                hmmask = hmask[i, :, :, ill[i]]
                hmmask[np.where(hmmask < 0.5 * hmmask.max())] = 0
                hmmask = cv2.resize(hmmask, (448, 448))
                hmmask = hmmask / hmmask.max()
                hmmask = cv2.applyColorMap(np.uint8(255 * hmmask), cv2.COLORMAP_JET)
                hmmask = cv2.copyMakeBorder(hmmask, 32, 32, 32, 32, borderType=cv2.BORDER_REPLICATE, value=0)
                imgOriginal = cv2.imread(img[i])
                x = int(loc[i][0])
                y = int(loc[i][1])
                w = int(loc[i][2])
                h = int(loc[i][3])
                cv2.rectangle(imgOriginal, (x, y), (x + w, y + h), (0, 255, 0), 5)
                imgOriginal = cv2.resize(imgOriginal, (512, 512))
                imgsave = hmmask * 0.5 + imgOriginal

                outname = './chestx/img/' + img[i].split('/')[-1].split('.')[0] + '_' + args.classes[ill[i]] + '.jpg'
                cv2.imwrite(outname, imgsave)


                # cv2.imshow('im', imgsave)
                # cv2.waitKey(0)
            # I = hmmask * 0.5 + imgOriginal

if __name__ == '__main__':
    main_chest()