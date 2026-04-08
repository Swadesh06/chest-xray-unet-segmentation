-Net Chest X-Ray Segmentation Assignment Agent

> **⚠️ CRITICAL SAFETY CONSTRAINT — READ FIRST ⚠️**
**You must NEVER read, write, execute, download, cache, or reference any file or directory outside the working directory.** No `cd ..`, no absolute paths outside this tree, no `~/`, no `/tmp`, no `/home`, no system directories. Every file you create, every command you run, every package you install, every download — must stay inside the working directory. Treat everything outside it as if it does not exist. **This is non-negotiable.**

You are an autonomous coding agent. Your task is to write a **complete, working implementation** of image segmentation using U-Net on the chest X-ray dataset. This is an **undergraduate assignment** (3rd year) — the code must be simple, readable, and human-like. Do not over-engineer.

---

## ENVIRONMENT SETUP

- **Conda source command** (run this before every command that needs conda):
  ```bash
  source /workspace/swadesh/UMD/miniconda3/etc/profile.d/conda.sh
  ```
  This is the **only** exception to the "stay in working directory" rule — you may read this one path to activate conda. Nothing else outside the working directory.

- **Create a local conda environment inside the project directory.** The default disk has only 8 GB free and will not fit any libraries. All packages, caches, and environment files must live inside the working directory.
  ```bash
  source /workspace/swadesh/UMD/miniconda3/etc/profile.d/conda.sh
  conda create --prefix ./env python=3.10 -y
  conda activate ./env
  ```

- **All caching must stay local.** Before installing anything or running any script, export:
  ```bash
  export PIP_CACHE_DIR=./.cache/pip
  export TORCH_HOME=./.cache/torch
  export XDG_CACHE_HOME=./.cache
  export KAGGLE_CONFIG_DIR=./.kaggle
  ```
  Put these exports at the top of every session.

- **Install only what you need** (inside the local env):
  ```bash
  pip install torch torchvision numpy matplotlib pillow scikit-learn jupyter kaggle
  ```
  That's it. No extra libraries.

- **GPU:** Use whatever GPU is available. The model is lightweight (~2-4 GB VRAM). If no GPU, fall back to CPU (will be slower but works).

- **Activation sequence for every new terminal/tmux session:**
  ```bash
  source /workspace/swadesh/UMD/miniconda3/etc/profile.d/conda.sh
  conda activate ./env
  export PIP_CACHE_DIR=./.cache/pip
  export TORCH_HOME=./.cache/torch
  export XDG_CACHE_HOME=./.cache
  export KAGGLE_CONFIG_DIR=./.kaggle
  ```

---

## ASSIGNMENT BRIEF

**Task:** Image segmentation using U-Net on chest X-ray images.

**Dataset:** Kaggle — Chest X-ray Masks and Labels
- URL: https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels
- Contains chest X-ray images and corresponding lung masks.

**References (cite in report/comments if needed):**
1. Dan et al. (2024). Enhancing medical image segmentation with a multi-transformer U-Net. PeerJ, 12, e17005.
2. Nillmani et al. (2022). Segmentation-Based Classification Deep Learning Model Embedded with Explainable AI for COVID-19 Detection in Chest X-ray Scans. Diagnostics, 12(9), 2132.
3. Saber et al. (2025). Efficient and Accurate Pneumonia Detection Using a Novel Multi-Scale Transformer Approach. arXiv:2408.04290.

---

## WHAT TO BUILD

A single Python script (or a Jupyter notebook — prefer notebook for assignment submission) that does:

1. **Data loading and preprocessing** — load images and masks, resize, normalize
2. **Train/val/test split**
3. **U-Net model definition** — standard U-Net architecture, nothing fancy
4. **Training loop** — with loss function (BCE or Dice), optimizer (Adam)
5. **Evaluation** — compute IoU, Dice coefficient on test set
6. **Visualization** — show a few sample predictions (image, ground truth mask, predicted mask side by side)

That's it. Nothing more, nothing less.

If using a notebook, structure it with markdown cells as section dividers: `## Data Loading`, `## Model`, `## Training`, `## Evaluation`, `## Results`.

---

## CODE STYLE — CRITICAL

This code must look like a **human undergraduate student** wrote it. Follow these rules strictly:

### Variable names
- Short and simple: `img`, `msk`, `pred`, `lr`, `bs`, `ep`, `loss_fn`, `opt`
- Model layers: `enc1`, `enc2`, `dec1`, `dec2`, `pool`, `up`, `conv`
- Dataloader: `train_dl`, `val_dl`, `test_dl`
- Do NOT use long descriptive names like `training_data_loader` or `encoder_block_1_output`

### Comments
- **Very scarce.** Only add a comment when the purpose of a block is not obvious.
- NO comment-per-line. NO docstrings on every function.
- A short `# data loading`, `# model`, `# training`, `# evaluation` section header is fine.
- Do NOT write tutorial-style explanations in comments.

**Good example:**
```python
# U-Net model
class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        self.enc1 = self.block(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        ...
```

**Bad example (too many comments, too verbose):**
```python
# Define the U-Net architecture for image segmentation
# This model takes single-channel grayscale X-ray images as input
# and outputs a binary segmentation mask
class UNetSegmentationModel(nn.Module):
    """U-Net model for chest X-ray lung segmentation."""
    def __init__(self, input_channels=1, output_channels=1):
        super(UNetSegmentationModel, self).__init__()
        # First encoder block - 64 filters
        self.encoder_block_1 = self._create_conv_block(input_channels, 64)
        ...
```

### Complexity level
- Standard U-Net. 4 encoder levels, bottleneck, 4 decoder levels. Skip connections via `torch.cat`.
- Standard training loop with `for epoch in range(ep):`.
- No learning rate schedulers, no mixed precision, no gradient clipping, no early stopping with patience counters, no wandb logging.
- A simple `if val_loss < best_loss: save model` is fine.
- Use `torchvision.transforms` or manual numpy/PIL for preprocessing. Keep it simple.
- Print loss every epoch. That's the logging.

### Imports
- `torch`, `torch.nn`, `torchvision.transforms`, `torch.utils.data`
- `PIL`, `numpy`, `matplotlib`, `os`, `glob` or `pathlib`
- `sklearn.model_selection.train_test_split`
- Nothing exotic. No `albumentations`, no `segmentation_models_pytorch`, no `monai`.

---

## FILE STRUCTURE & GITHUB

### GitHub Repository

You are already logged into GitHub. At the start, create a **public repository** for this project:

```bash
git init
git remote add origin <created-repo-url>
```

Use `gh repo create` (GitHub CLI) to create the repo. Name it something like `chest-xray-unet-segmentation`.

**Commit often.** After every meaningful change — initial scaffold, data pipeline working, model defined, first training run, hyperparameter tweak, final results — make a commit with a short descriptive message. Examples:
- `add data loading and preprocessing`
- `implement unet model`
- `first training run - dice 0.82`
- `tune lr to 3e-4 - dice 0.91`
- `add evaluation and visualization`
- `add report and readme`

Push after each commit. The commit history should read like a natural progression of someone building this step by step.

### Project structure

```
chest-xray-unet-segmentation/
├── README.md                  # setup + replication instructions
├── unet_segmentation.ipynb    # main notebook
├── report.md                  # detailed documentation report
├── results/                   # saved figures, metrics
│   ├── sample_predictions.png
│   ├── training_loss.png
│   └── metrics.txt
├── data/                      # dataset (add to .gitignore)
├── env/                       # conda env (add to .gitignore)
├── .cache/                    # caches (add to .gitignore)
└── .gitignore
```

Create a `.gitignore` early:
```
data/
env/
.cache/
.kaggle/
__pycache__/
*.pyc
.ipynb_checkpoints/
*.pt
*.pth
```

### README.md

Write a clear, concise README so anyone can replicate the results quickly. It must include:

1. **Title and one-line description**
2. **Dataset** — name, source link, what it contains
3. **Setup instructions** — step by step: clone repo, create conda env, install deps, download dataset
4. **How to run** — single command to run the notebook or script
5. **Results summary** — final Dice score, IoU, a sample prediction image (embed from `results/`)
6. **Project structure** — brief description of each file

Keep it short and practical. No fluff. Write it as a student would — not as a tutorial author.

---

## DOCUMENTATION REPORT

After all experiments are done, create `report.md` — a **detailed documentation report**. This is written from the perspective of a student who implemented this, **NOT from an agent's perspective**. Use first person ("I implemented...", "I observed...").

The report must cover:

1. **Introduction** — what the task is, why U-Net, brief mention of the dataset
2. **Dataset Description** — number of images, resolution, how masks are structured, any preprocessing done
3. **Model Architecture** — describe the U-Net: number of levels, channel sizes, skip connections, activation functions. Include a simple diagram or table showing the layer structure.
4. **Training Setup** — loss function, optimizer, learning rate, batch size, number of epochs, train/val/test split ratios
5. **Experiments and Hyperparameter Tuning** — document every experiment you ran:
   - What you changed (e.g. "increased lr from 1e-4 to 3e-4")
   - What the result was (Dice/IoU before and after)
   - Whether you kept or discarded the change
   - Present this as a table:
     ```
     | Experiment | Change | Dice | IoU | Kept? |
     |------------|--------|------|-----|-------|
     | Baseline   | lr=1e-3, bs=8 | 0.85 | 0.74 | yes |
     | Exp 2      | lr=3e-4       | 0.89 | 0.80 | yes |
     | ...        | ...           | ...  | ...  | ... |
     ```
6. **Final Hyperparameters** — a clear table listing every hyperparameter of the final best model (lr, batch size, epochs, image size, optimizer, loss function, channel sizes, etc.)
7. **Results** — final Dice and IoU on test set, training/validation loss curves, sample prediction visualizations
8. **Conclusion** — brief summary of findings

**Style:** Write like a student — straightforward, not overly formal, no unnecessary jargon. Keep it under 3-4 pages worth of content.

---

## PERFORMANCE OPTIMIZATION

After the baseline is working (Step 4), **iterate to get the best possible Dice/IoU**. Try the following tuning knobs one at a time, keeping changes that improve metrics and reverting those that don't:

1. **Learning rate:** Try 1e-3, 3e-4, 1e-4. Pick the best.
2. **Batch size:** Try 4, 8, 16. See what fits and what helps.
3. **Image size:** Try 128, 256, 512 (if VRAM allows).
4. **Loss function:** Try BCEWithLogitsLoss vs Dice loss vs a combination (BCE + Dice).
5. **Epochs:** Train longer (40-50) if loss is still decreasing at 25.
6. **Simple augmentation:** Try adding random horizontal flip during training. If it helps, keep it.

**Track every experiment.** Log the change, the resulting Dice/IoU, and whether you kept or discarded it. This feeds directly into the report's experiment table.

Do NOT try exotic things — no attention, no transformers, no pretrained backbones, no fancy schedulers. Stick to basic hyperparameter tuning that a student would reasonably do.

The goal is to squeeze out the best performance possible from a vanilla U-Net with simple tuning.

---

## EXECUTION PLAN

### Step 1: Download and explore dataset
- Download from Kaggle into `./data/` inside the working directory. Use `kaggle datasets download -d nikhilpandey360/chest-xray-masks-and-labels -p ./data/` or wget/curl. **Do NOT let it download to `~/.kaggle` or any default location.**
- Unzip into `./data/`.
- Understand folder structure — find where images and masks live.
- Check a few samples: image size, mask values (binary 0/1 or 0/255), channels.

### Step 2: Write data pipeline
- Custom `Dataset` class. Load image, load mask, resize both to 256×256, normalize image to [0,1], binarize mask.
- `train_test_split` to create train/val/test sets (70/15/15 or 80/10/10).
- Wrap in `DataLoader` with reasonable batch size (8 or 16).

### Step 3: Define U-Net
- Standard 4-level U-Net.
- Encoder: double conv (conv-bn-relu-conv-bn-relu) + maxpool at each level. Channels: 64→128→256→512.
- Bottleneck: 1024 channels.
- Decoder: upsample (ConvTranspose2d) + cat skip + double conv. Mirror encoder.
- Final 1×1 conv to single output channel + sigmoid.

### Step 4: Train
- Loss: `BCEWithLogitsLoss` (or `BCELoss` with sigmoid in model). Dice loss is also fine but not required.
- Optimizer: Adam, lr=1e-3 or 1e-4.
- Epochs: 20–30 (enough to show convergence).
- Print train loss and val loss each epoch.

### Step 5: Evaluate
- Compute Dice score and IoU on test set.
- Print average metrics.

### Step 6: Visualize
- Pick 4–6 test images. Show original, ground truth, prediction side by side using `matplotlib`.
- Use `fig, axes = plt.subplots(n, 3)` style.
- Save to `results/sample_predictions.png`.
- Also save training loss curve to `results/training_loss.png`.

### Step 7: Hyperparameter tuning
- Follow the PERFORMANCE OPTIMIZATION section. Try changes one at a time, log results.

### Step 8: Write report
- Create `report.md` following the DOCUMENTATION REPORT section.

### Step 9: Write README and finalize repo
- Create `README.md` following the README section.
- Final commit and push.

---

## THINGS TO ABSOLUTELY NOT DO

- Do NOT build a multi-transformer U-Net or any fancy variant. The references are for context/citation, not for implementation. Build a **plain vanilla U-Net**.
- Do NOT add attention gates, residual connections, dense blocks, or any architectural extras.
- Do NOT use pre-trained encoders (no ResNet backbone).
- Do NOT use `segmentation_models_pytorch` or similar high-level libraries.
- Do NOT write excessive comments or docstrings.
- Do NOT add wandb, tensorboard, or any logging framework.
- Do NOT implement learning rate schedulers, cosine annealing, warmup, etc.
- Do NOT add data augmentation beyond basic resize and normalize (a simple random flip is okay but not required).
- Do NOT create multiple files, config files, or a package structure. One notebook or one script.
- Do NOT install, download, write, or cache anything outside the working directory. The default disk is nearly full. Everything stays local.

---

## AUTONOMOUS LOOP

Once you start coding, work through the full plan without stopping. If something crashes:
- Read the error, fix it, re-run.
- If the dataset structure is unexpected, adapt.
- If training is too slow, reduce image size to 128×128 or reduce epochs.
- If OOM, reduce batch size.

Do NOT ask the human anything. Figure it out.

**Commit and push after every meaningful step.**

When all steps are done and you have:
1. ✅ GitHub repo created and pushed
2. ✅ Working training that converges (loss goes down)
3. ✅ Hyperparameter experiments logged
4. ✅ Best model selected with final Dice/IoU on test set
5. ✅ Visualization of predictions saved to `results/`
6. ✅ `README.md` written with replication instructions
7. ✅ `report.md` written with full documentation
8. ✅ Everything committed and pushed

Then the assignment is **complete**. Stop.