import numpy as np
from glob import glob
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import confusion_matrix
import time
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
# from UNetFormer_MMSAM_heatmap import UNetFormer as UNetFormer
from moadapter_heatmap  import M2fnet as M2fnet
# from FTUNetFormer import ft_unetformer as FTUNetFormer
try:
    from urllib.request import URLopener
except ImportError:
    from urllib import URLopener

if MODEL == 'UNetformer':
    net = M2fnet(num_classes=N_CLASSES).cuda()


params = 0
for name, param in net.named_parameters():
    params += param.nelement()
print('All Params:   ', params)

params1 = 0
params2 = 0
for name, param in net.image_encoder.named_parameters():
    if "Adapter" not in name:
    # if "lora_" not in name:
    # if "lora_" not in name and "Adapter" not in name:
        params1 += param.nelement()
    else:
        params2 += param.nelement()
print('ImgEncoder:   ', params1)
print('Adapter:       ', params2)
# print('Lora: ', params2)
# print('Adapter_Lora: ', params2)
print('Others: ', params-params1-params2)


# print(net)

print("training : ", train_ids)
print("testing : ", test_ids)
train_set = ISPRS_dataset(train_ids, cache=CACHE)
train_loader = torch.utils.data.DataLoader(train_set,batch_size=BATCH_SIZE)

base_lr = 0.01
params_dict = dict(net.named_parameters())
params = []
for key, value in params_dict.items():
    if '_D' in key:
        # Decoder weights are trained at the nominal learning rate
        params += [{'params':[value],'lr': base_lr}]
    else:
        # Encoder weights are trained at lr / 2 (we have VGG-16 weights as initialization)
        params += [{'params':[value],'lr': base_lr / 2}]

optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)
# We define the scheduler
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [25, 35, 45], gamma=0.1)


def test(net, test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=1, window_size=WINDOW_SIZE):
    # Use the network on the test set
    if DATASET == 'Potsdam':
        test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id))[:, :, :3], dtype='float32') for id in test_ids)
        # test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id))[:, :, (3, 0, 1, 2)][:, :, :3], dtype='float32') for id in test_ids)
    ## Vaihingen
    else:
        test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_irs = (1 / 255 * np.asarray(io.imread(DSM_FOLDER.format(id), as_gray=True), dtype='float32') for id in
                test_ids)
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
    index = 0
    for img,ir, gt, gt_e in tqdm(zip(test_images, test_irs, test_labels, eroded_labels), total=len(test_ids), leave=False):
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

            # gt_patches = [np.copy(gt[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
            gt_patches = [(np.copy(gt[ x:x + w, y:y + h])) for x, y, w, h in coords]
            gt_patches = np.asarray(gt_patches)
            # gt_patches = Variable(torch.from_numpy(gt_patches).cuda(), volatile=True)
            with torch.no_grad():
                gt_patches = torch.from_numpy(gt_patches).cuda()


            ir_patches = [np.copy(ir[x:x + w, y:y + h]) for x, y, w, h in coords]
            ir_patches = np.asarray(ir_patches)

            with torch.no_grad():
                ir_patches = torch.from_numpy(ir_patches).cuda()
            # Do the inference
            outs, heatmap1, heatmap2,heatmap3,heatmap4= net(image_patches, ir_patches, mode='Test')
            outs = outs.data.cpu().numpy()

            image_patches = np.asarray(255 * torch.squeeze(image_patches).cpu(), dtype='uint8').transpose((1, 2, 0))
            gt_patches = np.asarray(torch.squeeze(gt_patches).cpu().numpy(),dtype='uint8')
            heatmap1 = cv2.resize(heatmap1, (512, 512))
            # heatmap[heatmap < 0.7] = 0
            heatmap1 = np.uint8(255 * heatmap1)
            heatmap1 = cv2.applyColorMap(heatmap1, cv2.COLORMAP_JET)
            heatmap1 = heatmap1[:, :, (2, 1, 0)]



            heatmap2 = cv2.resize(heatmap2, (512, 512))
            # heatmap[heatmap < 0.7] = 0
            heatmap2 = np.uint8(255 * heatmap2)
            heatmap2 = cv2.applyColorMap(heatmap2, cv2.COLORMAP_JET)
            heatmap2 = heatmap2[:, :, (2, 1, 0)]

            heatmap3 = cv2.resize(heatmap3, (512, 512))
            # heatmap[heatmap < 0.7] = 0
            heatmap3 = np.uint8(255 * heatmap3)
            heatmap3 = cv2.applyColorMap(heatmap3, cv2.COLORMAP_JET)
            heatmap3 = heatmap3[:, :, (2, 1, 0)]

            heatmap4= cv2.resize(heatmap4, (512, 512))
            # heatmap[heatmap < 0.7] = 0
            heatmap4 = np.uint8(255 * heatmap4)
            heatmap4 = cv2.applyColorMap(heatmap4, cv2.COLORMAP_JET)
            heatmap4 = heatmap4[:, :, (2, 1, 0)]


            fig = plt.figure()
            fig.add_subplot(1, 6, 1)
            plt.imshow(image_patches)
            # plt.title('CFNet', y=-0.1)
            plt.axis('off')
            fig.add_subplot(1, 6, 2)
            plt.imshow(heatmap1)
            # heatmap_str = './CFNet_features' + str(featureid) + '.jpg'
            # cv2.imwrite(heatmap_str, heatmap1)

            plt.axis('off')
            fig.add_subplot(1, 6, 3)
            plt.imshow(heatmap2)
            # heatmap_str = './CFNet_features' + str(featureid+1) + '.jpg'
            # cv2.imwrite(heatmap_str, heatmap2)

            plt.axis('off')
            fig.add_subplot(1, 6, 4)
            plt.imshow(heatmap3)

            plt.axis('off')
            fig.add_subplot(1, 6, 5)
            plt.imshow(heatmap4)

            plt.axis('off')
            fig.add_subplot(1, 6, 6)
            plt.imshow(gt_patches)

            # clear_output()
            plt.axis('off')

            # plt.show()
            # plt.savefig('heatmap.png', dpi=1200)
            plt.savefig('./visual/heatmap'+str(index)+'.pdf', dpi=1200)
            index += 1

            # Fill in the results array
            for out, (x, y, w, h) in zip(outs, coords):
                out = out.transpose((1, 2, 0))
                pred[x:x + w, y:y + h] += out
            del (outs)

        pred = np.argmax(pred, axis=-1)
        all_preds.append(pred)
        all_gts.append(gt)
        clear_output()

    accuracy = metrics(np.concatenate([p.ravel() for p in all_preds]),
                       np.concatenate([p.ravel() for p in all_gts]).ravel())
    if all:
        return accuracy, all_preds, all_gts
    else:
        return accuracy


if MODE == 'Train':
    train(net, optimizer, epochs, scheduler, weights=WEIGHTS, save_epoch=save_epoch)

elif MODE == 'Test':
    if DATASET == 'MAWS':
        net.load_state_dict(torch.load('./resultsf/UNetformer_last-28_0.590114390339679'), strict=True)
        net.eval()
        MIoU, all_preds, all_gts = test(net, test_ids, all=True, stride=256)
        print("MIoU: ", MIoU)
        for p, id_ in zip(all_preds, test_ids):
            img = convert_to_color(p)
            io.imsave('./visual/{}_tile_{}.png'.format('base', id_), img)

    elif DATASET == 'Potsdam':
        net.load_state_dict(torch.load('./resultsp/UNetformer_CSadapter_epoch41_0.845948277063731'), strict=True)
        net.eval()
        MIoU, all_preds, all_gts = test(net, test_ids, all=True, stride=256)
        print("MIoU: ", MIoU)
        for p, id_ in zip(all_preds, test_ids):
            img = convert_to_color(p)
            io.imsave('./visual/{}_tile_{}.png'.format('base', id_), img)

    elif DATASET == 'Hunan':
        net.load_state_dict(torch.load('./resultsh/UNetformer_xiaorong_k3_epoch11_0.4848190941484355'), strict=False)
        net.eval()
        MIoU, all_preds, all_gts = test(net, test_ids, all=True, stride=256)
        print("MIoU: ", MIoU)
        for p, id_ in zip(all_preds, test_ids):
            img = convert_to_color(p)
            io.imsave('./visual/heat_{}_tile_{}.png'.format('base', id_), img)
