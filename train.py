import numpy as np
from glob import glob
from tqdm.notebook import tqdm
from sklearn.metrics import confusion_matrix
import time
import torch.nn.functional as F
import cv2
import itertools
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from utils import *
from torch.autograd import Variable
from IPython.display import clear_output
from M2fnet_model import M2fnet as M2FNet

try:
    from urllib.request import URLopener
except ImportError:
    from urllib import URLopener

net = M2FNet(num_classes=N_CLASSES).cuda()

print("training : ", len(train_ids))
print("testing : ", len(test_ids))
train_set = ISPRS_dataset(train_ids, cache=CACHE)
train_loader = torch.utils.data.DataLoader(train_set,batch_size=BATCH_SIZE)

base_lr = 0.01
optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)
# We define the scheduler
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [15, 25, 35], gamma=0.1)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [12, 17, 22], gamma=0.1)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [5, 7, 9], gamma=0.1)

def test(net, test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    # Use the network on the test set
    if DATASET == 'Potsdam':
        test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id))[:, :, :3], dtype='float32') for id in test_ids)
        # test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id))[:, :, (3, 0, 1, 2)][:, :, :3], dtype='float32') for id in test_ids)

    else:
        test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_irs = (1 / 255 *np.asarray(io.imread(DSM_FOLDER.format(id),as_gray=True), dtype='float32') for id in test_ids)
    test_labels = (np.asarray(io.imread(LABEL_FOLDER.format(id)), dtype='uint8') for id in test_ids)
    if DATASET == 'Hunan':
        eroded_labels = ((np.asarray(io.imread(ERODED_FOLDER.format(id)), dtype='int64')) for id in test_ids)
    elif DATASET == 'MAWS':
        eroded_labels = ((np.asarray(io.imread(ERODED_FOLDER.format(id)), dtype='int64')) for id in test_ids)
    elif DATASET == 'Potsdam':
        eroded_labels = (convert_from_color(io.imread(ERODED_FOLDER.format(id))) for id in test_ids)


    all_preds = []
    all_gts = []

    # Switch the network to inference mode
    with torch.no_grad():
        for img, ir,gt, gt_e in tqdm(zip(test_images, test_irs, test_labels, eroded_labels), total=len(test_ids), leave=False):
            pred = np.zeros(img.shape[:2] + (N_CLASSES,))

            total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size

            for i, coords in enumerate(
                    tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total,
                        leave=False)):
                # Build the tensor
                image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
                image_patches = np.asarray(image_patches)
                # image_patches = Variable(torch.from_numpy(image_patches).cuda(), volatile=True)
                with torch.no_grad():
                    image_patches = torch.from_numpy(image_patches).cuda()

                ir_patches = [np.copy(ir[x:x + w, y:y + h]) for x, y, w, h in coords]
                ir_patches = np.asarray(ir_patches)
                # dsm_patches = Variable(torch.from_numpy(dsm_patches).cuda(), volatile=True)
                with torch.no_grad():
                    ir_patches = torch.from_numpy(ir_patches).cuda()
                # Do the inference
                outs = net(image_patches, ir_patches,mode='Test')
                outs = outs.data.cpu().numpy()


                for out, (x, y, w, h) in zip(outs, coords):
                    out = out.transpose((1, 2, 0))
                    pred[x:x + w, y:y + h] += out
                del (outs)

            pred = np.argmax(pred, axis=-1)
            all_preds.append(pred)
            if DATASET == 'Potsdam':
                all_gts.append(gt_e)
            else:
                all_gts.append(gt)
            clear_output()

    if DATASET == 'Hunan':
        accuracy = metrics_loveda(np.concatenate([p.ravel() for p in all_preds]),
                        np.concatenate([p.ravel() for p in all_gts]).ravel())
    else:
        accuracy = metrics(np.concatenate([p.ravel() for p in all_preds]),
                        np.concatenate([p.ravel() for p in all_gts]).ravel())
    if all:
        return accuracy, all_preds, all_gts
    else:
        return accuracy



def train(net, optimizer, epochs, scheduler=None, weights=WEIGHTS, save_epoch=1):
    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    weights = weights.cuda()

    iter_ = 0
    MIoU_best = 0.00
    for e in range(1, epochs + 1):
        # if scheduler is not None:
        #     scheduler.step()
        net.train()
        start_time = time.time()
        for batch_idx, (data, ir, target) in enumerate(train_loader):
            # data, ir, target = Variable(data.cuda()), Variable(ir.cuda()), Variable(target.cuda())
            data, ir, target = data.cuda(), ir.cuda(), target.cuda()
            optimizer.zero_grad()
            output = net(data, ir, mode='Train')

            loss = loss_calc(output, target, weights)
            # loss = CrossEntropy2d(output, target, weight=weights)
            loss.backward()
            optimizer.step()
            losses[iter_] = loss.data
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_])

            if iter_ % 100 == 0:
                clear_output()
                rgb = np.asarray(255 * np.transpose(data.data.cpu().numpy()[0], (1, 2, 0)), dtype='uint8')
                pred = np.argmax(output.data.cpu().numpy()[0], axis=0)
                gt = target.data.cpu().numpy()[0]
                print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}'.format(
                    e, epochs, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.data, accuracy(pred, gt)))
            iter_ += 1

            del (data, target, loss)

        if scheduler is not None:
            scheduler.step()
        if e % save_epoch == 0:
            train_time = time.time()
            print("Training time: {:.3f} seconds".format(train_time - start_time))
            # We validate with the largest possible stride for faster computing
            net.eval()
            MIoU = test(net, test_ids, all=False, stride=Stride_Size)
            net.train()
            test_time = time.time()
            print("Test time: {:.3f} seconds".format(test_time - train_time))
            if MIoU > MIoU_best:
                if DATASET == 'MAWS':
                    torch.save(net.state_dict(), './results/{}_epoch_{}_{}'.format(MODEL, e, MIoU))
                elif DATASET == 'Potsdam':
                    torch.save(net.state_dict(), './resultsp/{}_potsdam_epoch{}_{}'.format(MODEL, e, MIoU))
                elif DATASET == 'Hunan':
                    torch.save(net.state_dict(), './resultsm/{}_mmhunan_{}_{}'.format(MODEL, e, MIoU))
                MIoU_best = MIoU
    print('MIoU_best: ', MIoU_best)



if MODE == 'Train':
    train(net, optimizer, epochs, scheduler, weights=WEIGHTS, save_epoch=save_epoch)

elif MODE == 'Test':
    if DATASET == 'MAWS':
        net.load_state_dict(torch.load('./last-59.pth'), strict=False)
        net.eval()
        MIoU, all_preds, all_gts = test(net, test_ids, all=True, stride=256)
        print("MIoU: ", MIoU)
        for p, id_ in zip(all_preds, test_ids):
            img = convert_to_color(p)
            io.imsave('./resultsf/{}_{}.png'.format('base', id_), img)

    elif DATASET == 'Potsdam':
        net.load_state_dict(torch.load('./resultsp/'), strict=False)
        net.eval()
        MIoU, all_preds, all_gts = test(net, test_ids, all=True, stride=128)
        print("MIoU: ", MIoU)
        for p, id_ in zip(all_preds, test_ids):
            img = convert_to_color(p)
            io.imsave('./resultsp/{}_{}.png'.format('base', id_), img)

    elif DATASET == 'Hunan':
        net.load_state_dict(torch.load('./resultsh/'), strict=False)
        net.eval()
        MIoU, all_preds, all_gts = test(net, test_ids, all=True, stride=256)
        print("MIoU: ", MIoU)
        for p, id_ in zip(all_preds, test_ids):
            img = convert_to_color(p)
            io.imsave('./resultsh/{}_{}.png'.format('base', id_), img)
