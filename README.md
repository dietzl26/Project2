# SML Project 2
This folder contains the student template of project 2


## Environment Setup
### GPU Cluster
Please refer to our [cluster guide](https://lec.inf.ethz.ch/sml/2026/exercises/Cluster-Tutorial.pdf) for instructions on how to set up the environment on the GPU cluster.

### Google Colab
For those who are running the project with Google Colab, we prepared the interface to run the code in the `Instructions_GoogleColab.ipynb`.

Please check out the installation guide on Moodle for this.
Make sure you upload the whole project2 folder (including the uncompressed datasets) to your Google Drive and follow the instructions in the `Instructions_GoogleColab.ipynb` to run the code.

### Local Installation
If you have a computer with a GPU, you might want to run the project locally. In this case, please set up an Anaconda environment running `python3.10`. Please check out the installation guide, e.g. from [Informatik II](https://lec.inf.ethz.ch/mavt/informatik2/2024/exercises/exercise1.pdf) for this.

If you are using Windows, we recommend to use either the VS code terminal or the Anaconda terminal, which is installed with Anaconda.

Please activate your project 2 environment by using:
```
conda activate <environment_name>
```
Then navigate to the folder containing the project files and run:
```
pip install --upgrade pip
pip install -r requirements.txt
```

**Note on PyTorch and CUDA:** The default `pip install torch torchvision` installs PyTorch with CUDA 12.x support on Linux/Windows (the wheel bundles the CUDA runtime). On macOS, a CPU-only version with MPS (Metal) support is installed. If you need a specific CUDA version, install PyTorch manually before the other requirements:
```bash
# For CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# For CPU-only (Linux/Windows):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```
See https://pytorch.org/get-started/locally/ for all available options.

If you require any additional packages, run:
```
pip install <package_name>
```

Make sure to extract all the data in the `./datasets` folder.

## Code Documentation

### Overview

The project trains a U-Net convolutional neural network to segment ETH mugs in images — a binary segmentation task where the model predicts, for every pixel, whether it belongs to a mug (1) or the background (0). The final output is a Kaggle submission CSV with run-length-encoded (RLE) predictions.

---

### File Overview

| File | Purpose |
|---|---|
| `model.py` | Configurable U-Net (variable depth and width) |
| `unet_simplistic.py` | Reusable building blocks + fixed-depth baseline U-Net |
| `eth_mugs_dataset.py` | PyTorch Dataset that loads images and masks |
| `train.py` | Full training loop with validation, checkpointing, and prediction |
| `predict.py` | Standalone inference script for an existing checkpoint |
| `utils.py` | IoU metric, RLE encoding, CSV writer |

---

### Model (`model.py` + `unet_simplistic.py`)

The architecture is a **U-Net**: an encoder–decoder with skip connections.

**Encoder** (downsampling path):
- `DoubleConv`: two Conv2d → BatchNorm → ReLU blocks. Used as the first encoder layer and inside every Down/Up block.
- `Down`: MaxPool2d (halves spatial size) followed by DoubleConv (doubles channels).
- The encoder has one `inc` (initial DoubleConv) plus `depth` Down blocks. For `base_channels=64, depth=4` the channel progression is: 3 → 64 → 128 → 256 → 512 → 512.

**Decoder** (upsampling path):
- `Up`: upsamples spatially, concatenates with the matching encoder skip connection, then applies DoubleConv to fuse them.
  - `bilinear=True` (default): bilinear `Upsample` (no learned parameters).
  - `bilinear=False`: `ConvTranspose2d` (learned upsampling).
- `OutConv`: final 1×1 convolution that maps to `n_classes=1` output channel (logit).
- The decoder mirrors the encoder: `depth` Up blocks, channels halving back to `base_channels`.

**Skip connections**: the output of each encoder level is saved and concatenated with the corresponding decoder level, giving the decoder both fine-grained spatial detail (from early encoder layers) and semantic context (from deep layers).

**Configurable parameters:**
| Argument | Default | Effect |
|---|---|---|
| `--base_channels` | 64 | Width of the network (channels at level 0) |
| `--depth` | 4 | Number of down/up levels |
| `--bilinear` | True | Bilinear vs. transposed-conv upsampling |
| `--dropout` | 0.0 | Dropout2d on the bottleneck |

The **baseline** (`--arch baseline`) uses a fixed 2-level `UNet` from `unet_simplistic.py` with 64 base channels.

---

### Dataset (`eth_mugs_dataset.py`)

`ETHMugsDataset` loads from `train_data/rgb/` (JPEG images) and `train_data/masks/` (PNG binary masks).

- Images are resized to `(252, 378)` with bilinear interpolation and converted to a `(3, H, W)` float tensor in `[0, 1]`.
- Masks are loaded, binarised (values > 1 are divided by 255), resized with nearest-neighbour interpolation, and returned as a `(1, H, W)` float tensor.
- In `mode="test"`, masks are absent; the dataset returns `(image_tensor, image_id_string)` instead.
- `image_ids` can be passed to use a specific split (used for train/val separation in `train.py`).

---

### Loss Function (`train.py`)

Training uses **ComboLoss** = 0.5 × BCE + 0.5 × Dice.

- **BCEWithLogitsLoss**: standard binary cross-entropy applied to raw logits (numerically stable).
- **DiceLoss**: `1 − (2·|P∩G| + ε) / (|P| + |G| + ε)` where P = sigmoid(logits) and G = ground-truth mask. Dice loss directly optimises the overlap metric and handles class imbalance well (mugs occupy a small fraction of pixels).

The `--bce_weight` argument (default 0.5) controls the balance between the two terms.

---

### Training Loop (`train.py`)

1. **Train/val split**: all image IDs are shuffled with a fixed seed and split `(1 − val_frac) / val_frac`. Default: 85 % train, 15 % val.

2. **Optimiser**: Adam with `lr=1e-3` and `weight_decay=1e-5`.

3. **LR schedule**: cosine annealing over `--epochs` (default 120). The learning rate smoothly decays from `lr` to ~0.

4. **Mixed precision** (CUDA only): `torch.amp.autocast` + `GradScaler` halves memory and speeds up training.

5. **Per epoch**:
   - Forward pass → compute ComboLoss → backprop → optimizer step.
   - Every `--val_frequency` epochs (default every 2), evaluate on the validation set with threshold 0.5 and print val IoU.
   - Save `last_epoch.pth` every epoch; overwrite `best_model.pth` whenever val IoU improves.

6. **Threshold tuning**: after training, the best checkpoint is reloaded and evaluated at 9 thresholds (0.30 → 0.70 in steps of 0.05) on the validation set. The threshold with the highest IoU is used for test-set prediction.

7. **Prediction** (`--predict`, default on): runs inference on `test_data/` and writes `submission.csv`.

---

### Evaluation metric (`utils.py`)

**Intersection over Union (IoU)** = (intersection + ε) / (union + ε).

Both `pred_mask` and `gt_mask` are integer arrays with values 0 or 1. Bitwise AND gives the intersection, bitwise OR gives the union. The ε prevents 0/0 on empty masks.

---

### Kaggle submission (`utils.py` + `predict.py`)

Predictions are encoded as **Run-Length Encoding (RLE)** in column-major (Fortran) order, which is the format Kaggle expects. Each mask is flattened column-by-column; the RLE records the start position and length of each run of 1s. This is stored in the `EncodedPixels` column alongside `ImageId`.

`predict.py` is a standalone script that loads any saved `.pth` checkpoint and writes `submission.csv` next to it, without rerunning training.

---

## Running Code
Please note that the script `train.py` takes arguments when you run them. These arguments are used when the script is carried out. The arguments to a python script can be specified in the following manner:
```bash
python train.py --<argument_1_name> <argument_1_value> --<argument_2_name> <argument_2_value>
```
`train.py` takes two arguments, namely the path to the datasets folder and where the training loop should save your model checkpoints.

For more information on the available arguments to these scripts, please run the following command:
```bash
python train.py -h
```

### GPU Cluster
Please refer to our [cluster guide](https://lec.inf.ethz.ch/sml/2026/exercises/Cluster-Tutorial.pdf) for instructions on how to run the code on the GPU cluster.

### Google Colab
Check out the `Instructions_GoogleColab.ipynb` for instructions on how to run the code in Google Colab.

### Local Installation
To run your solution locally, first make sure you have activated your conda environment. Then open a terminal and run the following command with your arguments to train the model:
```bash
python train.py <your_arguments_here>
```
