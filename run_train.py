import os
os.makedirs('results', exist_ok=True)
import matplotlib
matplotlib.use('Agg')
import glob
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from tqdm import tqdm

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {dev}')

img_dir = 'data/data/Lung Segmentation/CXR_png'
msk_dir = 'data/data/Lung Segmentation/masks'

imgs = sorted(glob.glob(os.path.join(img_dir, '*.png')))
msks = sorted(glob.glob(os.path.join(msk_dir, '*.png')))

msk_names = {os.path.basename(m): m for m in msks}
pairs = []
for ip in imgs:
    bn = os.path.basename(ip)
    candidates = [bn, bn.replace('.png', '_mask.png')]
    for c in candidates:
        if c in msk_names:
            pairs.append((ip, msk_names[c]))
            break

if len(pairs) == 0:
    img_names = {os.path.basename(i): i for i in imgs}
    for mp in msks:
        bn = os.path.basename(mp)
        cand = bn.replace('_mask', '')
        if cand in img_names:
            pairs.append((img_names[cand], mp))

print(f'Found {len(pairs)} image-mask pairs')
print(f'Example: {os.path.basename(pairs[0][0])} -> {os.path.basename(pairs[0][1])}')

sample_img = np.array(Image.open(pairs[0][0]))
sample_msk = np.array(Image.open(pairs[0][1]))
print(f'Image shape: {sample_img.shape}, dtype: {sample_img.dtype}, range: [{sample_img.min()}, {sample_img.max()}]')
print(f'Mask shape: {sample_msk.shape}, dtype: {sample_msk.dtype}, unique vals: {np.unique(sample_msk)}')

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for i in range(3):
    im = np.array(Image.open(pairs[i][0]).convert('L'))
    mk = np.array(Image.open(pairs[i][1]).convert('L'))
    axes[0, i].imshow(im, cmap='gray')
    axes[0, i].set_title('X-ray')
    axes[0, i].axis('off')
    axes[1, i].imshow(mk, cmap='gray')
    axes[1, i].set_title('Mask')
    axes[1, i].axis('off')
plt.tight_layout()
plt.show()

class XrayDataset(Dataset):
    def __init__(self, pairs, img_sz=256, aug=False):
        self.pairs = pairs
        self.img_sz = img_sz
        self.aug = aug

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ip, mp = self.pairs[idx]
        img = Image.open(ip).convert('L').resize((self.img_sz, self.img_sz))
        msk = Image.open(mp).convert('L').resize((self.img_sz, self.img_sz), Image.NEAREST)

        if self.aug and np.random.rand() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            msk = msk.transpose(Image.FLIP_LEFT_RIGHT)

        img = np.array(img, dtype=np.float32) / 255.0
        msk = np.array(msk, dtype=np.float32)
        msk = (msk > 127).astype(np.float32)

        img = torch.from_numpy(img).unsqueeze(0)
        msk = torch.from_numpy(msk).unsqueeze(0)
        return img, msk

def make_loaders(pairs, img_sz=256, bs=8, aug=False):
    trn_p, tmp_p = train_test_split(pairs, test_size=0.3, random_state=42)
    val_p, tst_p = train_test_split(tmp_p, test_size=0.5, random_state=42)
    print(f'Split: train={len(trn_p)}, val={len(val_p)}, test={len(tst_p)}')

    trn_ds = XrayDataset(trn_p, img_sz, aug=aug)
    val_ds = XrayDataset(val_p, img_sz)
    tst_ds = XrayDataset(tst_p, img_sz)

    trn_dl = DataLoader(trn_ds, batch_size=bs, shuffle=True, num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)
    tst_dl = DataLoader(tst_ds, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)
    return trn_dl, val_dl, tst_dl, tst_p

class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        self.enc1 = self._block(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self._block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self._block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = self._block(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = self._block(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self._block(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self._block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self._block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self._block(128, 64)

        self.out_conv = nn.Conv2d(64, out_ch, 1)

    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        b = self.bottleneck(self.pool4(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)

model = UNet().to(dev)
total_params = sum(p.numel() for p in model.parameters())
print(f'UNet params: {total_params/1e6:.2f}M')

def dice_coeff(pred, tgt, smooth=1.0):
    pf = pred.view(-1)
    tf = tgt.view(-1)
    inter = (pf * tf).sum()
    return (2.0 * inter + smooth) / (pf.sum() + tf.sum() + smooth)

def iou_score(pred, tgt, smooth=1.0):
    pf = pred.view(-1)
    tf = tgt.view(-1)
    inter = (pf * tf).sum()
    union = pf.sum() + tf.sum() - inter
    return (inter + smooth) / (union + smooth)

class DiceLoss(nn.Module):
    def forward(self, pred, tgt, smooth=1.0):
        pred = torch.sigmoid(pred)
        pf = pred.view(-1)
        tf = tgt.view(-1)
        inter = (pf * tf).sum()
        return 1 - (2.0 * inter + smooth) / (pf.sum() + tf.sum() + smooth)

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, pred, tgt):
        return self.bce(pred, tgt) + self.dice(pred, tgt)

def train_model(model, trn_dl, val_dl, loss_fn, opt, ep=25, name='model'):
    trn_losses, val_losses = [], []
    best_vloss = float('inf')

    for e in range(ep):
        model.train()
        run_loss = 0.0
        pbar = tqdm(trn_dl, desc=f'Ep {e+1}/{ep} [trn]', leave=False)
        for img, msk in pbar:
            img, msk = img.to(dev), msk.to(dev)
            opt.zero_grad()
            out = model(img)
            loss = loss_fn(out, msk)
            loss.backward()
            opt.step()
            run_loss += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.4f}')
        trn_loss = run_loss / len(trn_dl)
        trn_losses.append(trn_loss)

        model.eval()
        run_vloss = 0.0
        with torch.no_grad():
            for img, msk in val_dl:
                img, msk = img.to(dev), msk.to(dev)
                out = model(img)
                run_vloss += loss_fn(out, msk).item()
        vloss = run_vloss / len(val_dl)
        val_losses.append(vloss)

        print(f'Ep {e+1}/{ep} - trn_loss: {trn_loss:.4f}, val_loss: {vloss:.4f}')

        if vloss < best_vloss:
            best_vloss = vloss
            torch.save(model.state_dict(), f'best_{name}.pt')

    return trn_losses, val_losses

def evaluate(model, tst_dl):
    model.eval()
    dices, ious = [], []
    with torch.no_grad():
        for img, msk in tqdm(tst_dl, desc='Eval'):
            img, msk = img.to(dev), msk.to(dev)
            out = torch.sigmoid(model(img))
            pred = (out > 0.5).float()
            for i in range(pred.shape[0]):
                dices.append(dice_coeff(pred[i], msk[i]).item())
                ious.append(iou_score(pred[i], msk[i]).item())
    avg_d = np.mean(dices)
    avg_iou = np.mean(ious)
    print(f'Test Dice: {avg_d:.4f}, Test IoU: {avg_iou:.4f}')
    return avg_d, avg_iou

exp_log = []

def log_exp(name, params, dice, iou, kept, trn_l=None, val_l=None):
    entry = {'name': name, 'params': params, 'dice': round(dice, 4),
             'iou': round(iou, 4), 'kept': kept}
    if trn_l is not None:
        entry['trn_losses'] = [round(x, 4) for x in trn_l]
        entry['val_losses'] = [round(x, 4) for x in val_l]
    exp_log.append(entry)
    print(f'[LOG] {name}: Dice={dice:.4f}, IoU={iou:.4f}, Kept={kept}')

def save_exp_log():
    os.makedirs('results', exist_ok=True)
    with open('results/metrics.txt', 'w') as f:
        f.write(f'{"Experiment":<20} {"Change":<35} {"Dice":<8} {"IoU":<8} {"Kept"}\n')
        f.write('-' * 80 + '\n')
        for ex in exp_log:
            f.write(f'{ex["name"]:<20} {ex["params"]:<35} {ex["dice"]:<8} {ex["iou"]:<8} {ex["kept"]}\n')
    with open('results/exp_log.json', 'w') as f:
        json.dump(exp_log, f, indent=2)
    print('Experiment log saved to results/')

LR = 1e-3
BS = 8
IMG_SZ = 256
EP = 25

trn_dl, val_dl, tst_dl, tst_pairs = make_loaders(pairs, img_sz=IMG_SZ, bs=BS)

model = UNet().to(dev)
loss_fn = nn.BCEWithLogitsLoss()
opt = torch.optim.Adam(model.parameters(), lr=LR)

print(f'Baseline: lr={LR}, bs={BS}, img_sz={IMG_SZ}, loss=BCE, ep={EP}')
trn_l, val_l = train_model(model, trn_dl, val_dl, loss_fn, opt, ep=EP, name='baseline')

model.load_state_dict(torch.load('best_baseline.pt', weights_only=True))
d_base, iou_base = evaluate(model, tst_dl)
log_exp('Baseline', f'lr={LR}, bs={BS}, BCE', d_base, iou_base, 'yes', trn_l, val_l)

best_dice = d_base
best_iou = iou_base
best_name = 'baseline'
best_trn_l = trn_l
best_val_l = val_l

# Exp 2: lr=3e-4
print('--- Exp 2: lr=3e-4 ---')
model = UNet().to(dev)
opt = torch.optim.Adam(model.parameters(), lr=3e-4)
loss_fn = nn.BCEWithLogitsLoss()
trn_l2, val_l2 = train_model(model, trn_dl, val_dl, loss_fn, opt, ep=EP, name='exp2')
model.load_state_dict(torch.load('best_exp2.pt', weights_only=True))
d2, iou2 = evaluate(model, tst_dl)

kept = d2 > best_dice
log_exp('Exp2-lr3e4', 'lr=3e-4', d2, iou2, 'yes' if kept else 'no', trn_l2, val_l2)
if kept:
    best_dice, best_iou, best_name = d2, iou2, 'exp2'
    best_trn_l, best_val_l = trn_l2, val_l2
    print(f'New best: Dice={best_dice:.4f}')

# Exp 3: lr=1e-4
print('--- Exp 3: lr=1e-4 ---')
model = UNet().to(dev)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.BCEWithLogitsLoss()
trn_l3, val_l3 = train_model(model, trn_dl, val_dl, loss_fn, opt, ep=EP, name='exp3')
model.load_state_dict(torch.load('best_exp3.pt', weights_only=True))
d3, iou3 = evaluate(model, tst_dl)

kept = d3 > best_dice
log_exp('Exp3-lr1e4', 'lr=1e-4', d3, iou3, 'yes' if kept else 'no', trn_l3, val_l3)
if kept:
    best_dice, best_iou, best_name = d3, iou3, 'exp3'
    best_trn_l, best_val_l = trn_l3, val_l3
    print(f'New best: Dice={best_dice:.4f}')

# Exp 4: bs=16
print('--- Exp 4: bs=16 ---')
cur_lr = 3e-4 if best_name == 'exp2' else (1e-4 if best_name == 'exp3' else 1e-3)

trn_dl16, val_dl16, tst_dl16, _ = make_loaders(pairs, img_sz=IMG_SZ, bs=16)
model = UNet().to(dev)
opt = torch.optim.Adam(model.parameters(), lr=cur_lr)
loss_fn = nn.BCEWithLogitsLoss()
trn_l4, val_l4 = train_model(model, trn_dl16, val_dl16, loss_fn, opt, ep=EP, name='exp4')
model.load_state_dict(torch.load('best_exp4.pt', weights_only=True))
d4, iou4 = evaluate(model, tst_dl16)

kept = d4 > best_dice
log_exp('Exp4-bs16', f'bs=16, lr={cur_lr}', d4, iou4, 'yes' if kept else 'no', trn_l4, val_l4)
if kept:
    best_dice, best_iou, best_name = d4, iou4, 'exp4'
    best_trn_l, best_val_l = trn_l4, val_l4
    print(f'New best: Dice={best_dice:.4f}')

# Exp 5: bs=4
print('--- Exp 5: bs=4 ---')
trn_dl4, val_dl4, tst_dl4, _ = make_loaders(pairs, img_sz=IMG_SZ, bs=4)
model = UNet().to(dev)
opt = torch.optim.Adam(model.parameters(), lr=cur_lr)
loss_fn = nn.BCEWithLogitsLoss()
trn_l5, val_l5 = train_model(model, trn_dl4, val_dl4, loss_fn, opt, ep=EP, name='exp5')
model.load_state_dict(torch.load('best_exp5.pt', weights_only=True))
d5, iou5 = evaluate(model, tst_dl4)

kept = d5 > best_dice
log_exp('Exp5-bs4', f'bs=4, lr={cur_lr}', d5, iou5, 'yes' if kept else 'no', trn_l5, val_l5)
if kept:
    best_dice, best_iou, best_name = d5, iou5, 'exp5'
    best_trn_l, best_val_l = trn_l5, val_l5
    print(f'New best: Dice={best_dice:.4f}')

# Exp 6: BCE+Dice combined loss
print('--- Exp 6: BCE+Dice loss ---')
cur_bs = 16 if best_name == 'exp4' else (4 if best_name == 'exp5' else 8)
trn_dlx, val_dlx, tst_dlx, _ = make_loaders(pairs, img_sz=IMG_SZ, bs=cur_bs)

model = UNet().to(dev)
opt = torch.optim.Adam(model.parameters(), lr=cur_lr)
loss_fn = BCEDiceLoss()
trn_l6, val_l6 = train_model(model, trn_dlx, val_dlx, loss_fn, opt, ep=EP, name='exp6')
model.load_state_dict(torch.load('best_exp6.pt', weights_only=True))
d6, iou6 = evaluate(model, tst_dlx)

kept = d6 > best_dice
log_exp('Exp6-BCEDice', f'BCE+Dice, lr={cur_lr}, bs={cur_bs}', d6, iou6, 'yes' if kept else 'no', trn_l6, val_l6)
if kept:
    best_dice, best_iou, best_name = d6, iou6, 'exp6'
    best_trn_l, best_val_l = trn_l6, val_l6
    print(f'New best: Dice={best_dice:.4f}')

# Exp 7: augmentation (horizontal flip)
print('--- Exp 7: horizontal flip augmentation ---')
cur_loss_fn = BCEDiceLoss() if best_name == 'exp6' else nn.BCEWithLogitsLoss()
cur_loss_name = 'BCE+Dice' if best_name == 'exp6' else 'BCE'

trn_pa, tmp_pa = train_test_split(pairs, test_size=0.3, random_state=42)
val_pa, tst_pa = train_test_split(tmp_pa, test_size=0.5, random_state=42)
trn_ds_aug = XrayDataset(trn_pa, IMG_SZ, aug=True)
val_ds_aug = XrayDataset(val_pa, IMG_SZ)
tst_ds_aug = XrayDataset(tst_pa, IMG_SZ)
trn_dl_aug = DataLoader(trn_ds_aug, batch_size=cur_bs, shuffle=True, num_workers=2, pin_memory=True)
val_dl_aug = DataLoader(val_ds_aug, batch_size=cur_bs, shuffle=False, num_workers=2, pin_memory=True)
tst_dl_aug = DataLoader(tst_ds_aug, batch_size=cur_bs, shuffle=False, num_workers=2, pin_memory=True)
print(f'Split: train={len(trn_pa)}, val={len(val_pa)}, test={len(tst_pa)}')

model = UNet().to(dev)
opt = torch.optim.Adam(model.parameters(), lr=cur_lr)
trn_l7, val_l7 = train_model(model, trn_dl_aug, val_dl_aug, cur_loss_fn, opt, ep=EP, name='exp7')
model.load_state_dict(torch.load('best_exp7.pt', weights_only=True))
d7, iou7 = evaluate(model, tst_dl_aug)

kept = d7 > best_dice
log_exp('Exp7-aug', f'h-flip, {cur_loss_name}, lr={cur_lr}, bs={cur_bs}', d7, iou7, 'yes' if kept else 'no', trn_l7, val_l7)
if kept:
    best_dice, best_iou, best_name = d7, iou7, 'exp7'
    best_trn_l, best_val_l = trn_l7, val_l7
    print(f'New best: Dice={best_dice:.4f}')

# Exp 8: 40 epochs with best config
print('--- Exp 8: 40 epochs ---')
use_aug = best_name == 'exp7'
cur_loss_fn = BCEDiceLoss() if ('exp6' in best_name or best_name == 'exp7') else nn.BCEWithLogitsLoss()

if use_aug:
    trn_dl_f, val_dl_f, tst_dl_f = trn_dl_aug, val_dl_aug, tst_dl_aug
else:
    trn_dl_f, val_dl_f, tst_dl_f, _ = make_loaders(pairs, img_sz=IMG_SZ, bs=cur_bs)

model = UNet().to(dev)
opt = torch.optim.Adam(model.parameters(), lr=cur_lr)
trn_l8, val_l8 = train_model(model, trn_dl_f, val_dl_f, cur_loss_fn, opt, ep=40, name='exp8')
model.load_state_dict(torch.load('best_exp8.pt', weights_only=True))
d8, iou8 = evaluate(model, tst_dl_f)

kept = d8 > best_dice
log_exp('Exp8-40ep', '40 epochs, best config', d8, iou8, 'yes' if kept else 'no', trn_l8, val_l8)
if kept:
    best_dice, best_iou, best_name = d8, iou8, 'exp8'
    best_trn_l, best_val_l = trn_l8, val_l8
    print(f'New best: Dice={best_dice:.4f}')

print(f'\nBest model: {best_name}, Dice={best_dice:.4f}, IoU={best_iou:.4f}')
save_exp_log()

model = UNet().to(dev)
model.load_state_dict(torch.load(f'best_{best_name}.pt', weights_only=True))
model.eval()

final_d, final_iou = evaluate(model, tst_dl_f)
print(f'Final - Dice: {final_d:.4f}, IoU: {final_iou:.4f}')

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.plot(best_trn_l, label='Train Loss')
ax.plot(best_val_l, label='Val Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training and Validation Loss')
ax.legend()
plt.tight_layout()
plt.savefig('results/training_loss.png', dpi=150)
plt.show()
print('Saved training_loss.png')

n_show = 6
fig, axes = plt.subplots(n_show, 3, figsize=(10, n_show * 3))
axes[0, 0].set_title('X-ray')
axes[0, 1].set_title('Ground Truth')
axes[0, 2].set_title('Prediction')

model.eval()
shown = 0
with torch.no_grad():
    for img, msk in tst_dl_f:
        img_d = img.to(dev)
        out = torch.sigmoid(model(img_d)).cpu()
        pred = (out > 0.5).float()
        for i in range(img.shape[0]):
            if shown >= n_show:
                break
            axes[shown, 0].imshow(img[i, 0], cmap='gray')
            axes[shown, 0].axis('off')
            axes[shown, 1].imshow(msk[i, 0], cmap='gray')
            axes[shown, 1].axis('off')
            axes[shown, 2].imshow(pred[i, 0], cmap='gray')
            axes[shown, 2].axis('off')
            shown += 1
        if shown >= n_show:
            break

plt.tight_layout()
plt.savefig('results/sample_predictions.png', dpi=150)
plt.show()
print('Saved sample_predictions.png')

print('--- Experiment Summary ---')
print(f'{"Experiment":<20} {"Dice":<10} {"IoU":<10} {"Kept"}')
print('-' * 50)
for ex in exp_log:
    print(f'{ex["name"]:<20} {ex["dice"]:<10} {ex["iou"]:<10} {ex["kept"]}')
print(f'\nBest: {best_name} - Dice={best_dice:.4f}, IoU={best_iou:.4f}')
