import os, glob, json, numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

class XrayDS(Dataset):
    def __init__(self, prs, sz=256):
        self.prs, self.sz = prs, sz
    def __len__(self): return len(self.prs)
    def __getitem__(self, idx):
        ip, mp = self.prs[idx]
        img = np.array(Image.open(ip).convert('L').resize((self.sz, self.sz)), dtype=np.float32) / 255.0
        msk = (np.array(Image.open(mp).convert('L').resize((self.sz, self.sz), Image.NEAREST), dtype=np.float32) > 127).astype(np.float32)
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

# load best model
model = UNet().to(dev)
model.load_state_dict(torch.load('best_exp8.pt', weights_only=True))
model.eval()

# test set
_, tmp = train_test_split(pairs, test_size=0.3, random_state=42)
_, tst_p = train_test_split(tmp, test_size=0.5, random_state=42)
tst_dl = DataLoader(XrayDS(tst_p), batch_size=8, shuffle=False, num_workers=2)

# sample predictions
n = 6
fig, axes = plt.subplots(n, 3, figsize=(10, n * 3))
axes[0, 0].set_title('X-ray')
axes[0, 1].set_title('Ground Truth')
axes[0, 2].set_title('Prediction')
shown = 0
with torch.no_grad():
    for img, msk in tst_dl:
        out = torch.sigmoid(model(img.to(dev))).cpu()
        pred = (out > 0.5).float()
        for i in range(img.shape[0]):
            if shown >= n: break
            axes[shown, 0].imshow(img[i, 0], cmap='gray'); axes[shown, 0].axis('off')
            axes[shown, 1].imshow(msk[i, 0], cmap='gray'); axes[shown, 1].axis('off')
            axes[shown, 2].imshow(pred[i, 0], cmap='gray'); axes[shown, 2].axis('off')
            shown += 1
        if shown >= n: break
plt.tight_layout()
plt.savefig('results/sample_predictions.png', dpi=150)
print('Saved sample_predictions.png')

# training loss curve from best model
with open('results/exp8.json') as f:
    best = json.load(f)
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.plot(best['trn_losses'], label='Train Loss')
ax.plot(best['val_losses'], label='Val Loss')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.set_title('Training and Validation Loss (Best Model - Exp8)')
ax.legend()
plt.tight_layout()
plt.savefig('results/training_loss.png', dpi=150)
print('Saved training_loss.png')

# collect all results into metrics.txt
all_res = []
# baseline + exp2 from sequential run
all_res.append({'name': 'Baseline', 'params': 'lr=1e-3, bs=8, BCE, 25ep', 'dice': 0.9630, 'iou': 0.9299, 'kept': 'yes'})
all_res.append({'name': 'Exp2-lr3e4', 'params': 'lr=3e-4', 'dice': 0.9637, 'iou': 0.9311, 'kept': 'yes'})

exp_files = sorted(glob.glob('results/exp*.json'))
for ef in exp_files:
    with open(ef) as f:
        d = json.load(f)
    nm = d['name']
    params = f"lr={d['lr']}, bs={d['bs']}, {d['loss']}, {'aug' if d['aug'] else 'no-aug'}, {d['ep']}ep"
    all_res.append({'name': nm, 'params': params, 'dice': d['dice'], 'iou': d['iou'], 'kept': '-'})

# mark best
best_d = max(r['dice'] for r in all_res)
for r in all_res:
    if r['dice'] == best_d:
        r['kept'] = 'BEST'

with open('results/metrics.txt', 'w') as f:
    f.write(f'{"Experiment":<20} {"Config":<45} {"Dice":<10} {"IoU":<10} {"Status"}\n')
    f.write('-' * 95 + '\n')
    for r in all_res:
        f.write(f'{r["name"]:<20} {r["params"]:<45} {r["dice"]:<10} {r["iou"]:<10} {r["kept"]}\n')
print('Saved metrics.txt')

# combined json
with open('results/exp_log.json', 'w') as f:
    json.dump(all_res, f, indent=2)
print('Saved exp_log.json')
print('Done generating all visualizations')
