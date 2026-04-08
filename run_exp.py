import os
os.makedirs('results', exist_ok=True)
import matplotlib
matplotlib.use('Agg')
import sys
import glob
import json
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- data ---
img_dir = 'data/data/Lung Segmentation/CXR_png'
msk_dir = 'data/data/Lung Segmentation/masks'
imgs = sorted(glob.glob(os.path.join(img_dir, '*.png')))
msks = sorted(glob.glob(os.path.join(msk_dir, '*.png')))
msk_names = {os.path.basename(m): m for m in msks}
pairs = []
for ip in imgs:
    bn = os.path.basename(ip)
    for c in [bn, bn.replace('.png', '_mask.png')]:
        if c in msk_names:
            pairs.append((ip, msk_names[c]))
            break
if len(pairs) == 0:
    img_names = {os.path.basename(i): i for i in imgs}
    for mp in msks:
        cand = os.path.basename(mp).replace('_mask', '')
        if cand in img_names:
            pairs.append((img_names[cand], mp))

class XrayDS(Dataset):
    def __init__(self, pairs, sz=256, aug=False):
        self.pairs, self.sz, self.aug = pairs, sz, aug
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        ip, mp = self.pairs[idx]
        img = Image.open(ip).convert('L').resize((self.sz, self.sz))
        msk = Image.open(mp).convert('L').resize((self.sz, self.sz), Image.NEAREST)
        if self.aug and np.random.rand() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            msk = msk.transpose(Image.FLIP_LEFT_RIGHT)
        img = np.array(img, dtype=np.float32) / 255.0
        msk = (np.array(msk, dtype=np.float32) > 127).astype(np.float32)
        return torch.from_numpy(img).unsqueeze(0), torch.from_numpy(msk).unsqueeze(0)

class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        self.enc1 = self._blk(in_ch, 64); self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self._blk(64, 128);   self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self._blk(128, 256);  self.pool3 = nn.MaxPool2d(2)
        self.enc4 = self._blk(256, 512);  self.pool4 = nn.MaxPool2d(2)
        self.bot = self._blk(512, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2); self.dec4 = self._blk(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2);  self.dec3 = self._blk(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2);  self.dec2 = self._blk(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2);   self.dec1 = self._blk(128, 64)
        self.out_conv = nn.Conv2d(64, out_ch, 1)
    def _blk(self, ic, oc):
        return nn.Sequential(nn.Conv2d(ic, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(True),
                             nn.Conv2d(oc, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(True))
    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2)); e4 = self.enc4(self.pool3(e3))
        b = self.bot(self.pool4(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.out_conv(d1)

def dice_coeff(p, t, s=1.0):
    pf, tf = p.view(-1), t.view(-1)
    i = (pf * tf).sum()
    return (2*i + s) / (pf.sum() + tf.sum() + s)

def iou_score(p, t, s=1.0):
    pf, tf = p.view(-1), t.view(-1)
    i = (pf * tf).sum()
    return (i + s) / (pf.sum() + tf.sum() - i + s)

class DiceLoss(nn.Module):
    def forward(self, p, t, s=1.0):
        p = torch.sigmoid(p); pf, tf = p.view(-1), t.view(-1)
        i = (pf * tf).sum()
        return 1 - (2*i + s) / (pf.sum() + tf.sum() + s)

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(); self.dice = DiceLoss()
    def forward(self, p, t): return self.bce(p, t) + self.dice(p, t)

def run_exp(name, lr, bs, ep, loss_type, img_sz, aug):
    print(f'[{name}] Starting: lr={lr}, bs={bs}, ep={ep}, loss={loss_type}, sz={img_sz}, aug={aug}')
    trn_p, tmp_p = train_test_split(pairs, test_size=0.3, random_state=42)
    val_p, tst_p = train_test_split(tmp_p, test_size=0.5, random_state=42)
    trn_dl = DataLoader(XrayDS(trn_p, img_sz, aug), batch_size=bs, shuffle=True, num_workers=2, pin_memory=True)
    val_dl = DataLoader(XrayDS(val_p, img_sz), batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)
    tst_dl = DataLoader(XrayDS(tst_p, img_sz), batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)

    model = UNet().to(dev)
    loss_fn = BCEDiceLoss() if loss_type == 'bce_dice' else nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_vl = float('inf')
    trn_ls, val_ls = [], []

    for e in range(ep):
        model.train(); rl = 0.0
        for img, msk in trn_dl:
            img, msk = img.to(dev), msk.to(dev)
            opt.zero_grad(); out = model(img); loss = loss_fn(out, msk)
            loss.backward(); opt.step(); rl += loss.item()
        tl = rl / len(trn_dl); trn_ls.append(tl)

        model.eval(); rv = 0.0
        with torch.no_grad():
            for img, msk in val_dl:
                img, msk = img.to(dev), msk.to(dev)
                rv += loss_fn(model(img), msk).item()
        vl = rv / len(val_dl); val_ls.append(vl)
        print(f'[{name}] Ep {e+1}/{ep} - trn: {tl:.4f}, val: {vl:.4f}')
        if vl < best_vl:
            best_vl = vl; torch.save(model.state_dict(), f'best_{name}.pt')

    model.load_state_dict(torch.load(f'best_{name}.pt', weights_only=True))
    model.eval(); dices, ious = [], []
    with torch.no_grad():
        for img, msk in tst_dl:
            img, msk = img.to(dev), msk.to(dev)
            pred = (torch.sigmoid(model(img)) > 0.5).float()
            for i in range(pred.shape[0]):
                dices.append(dice_coeff(pred[i], msk[i]).item())
                ious.append(iou_score(pred[i], msk[i]).item())
    avg_d, avg_iou = np.mean(dices), np.mean(ious)
    print(f'[{name}] DONE - Dice: {avg_d:.4f}, IoU: {avg_iou:.4f}')

    res = {'name': name, 'lr': lr, 'bs': bs, 'ep': ep, 'loss': loss_type,
           'img_sz': img_sz, 'aug': aug, 'dice': round(avg_d, 4), 'iou': round(avg_iou, 4),
           'trn_losses': [round(x, 4) for x in trn_ls],
           'val_losses': [round(x, 4) for x in val_ls]}
    with open(f'results/{name}.json', 'w') as f:
        json.dump(res, f, indent=2)
    return res

if __name__ == '__main__':
    cfg = json.loads(sys.argv[1])
    run_exp(**cfg)
