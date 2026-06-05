"""
Code template for training a unet model on the ETHMugs dataset.
This is intended merely as an example of how to run a training on the ISG-INF cluster.
"""

import argparse
import os
from datetime import datetime

import torch
from PIL import Image

from eth_mugs_dataset_simplistic import ETHMugsDataset
from utils import compute_iou, save_predictions
from unet_simplistic import UNet


def build_model():
    #  Add your model definition here
    """Build the model."""
    return UNet(3, 1)


def train(
    ckpt_dir: str,
    train_data_root: str,
    test_data_root: str,
):
    """Train function."""
    # Logging and validation settings
    train_batch_size = 8
    test_batch_size = 1

    # Set your own values for the hyperparameters
    num_epochs = 10
    lr = 1e-4

    print(f"[INFO]: Number of training epochs: {num_epochs}")

    # Choose Device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Define your Dataset and DataLoader
    # ETHMugsDataset 
    # Data loaders
    train_dataset = ETHMugsDataset(root_dir=train_data_root, mode="train")
    test_dataset = ETHMugsDataset(root_dir=test_data_root, mode="test")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    out_dir = os.path.join('prediction')
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO]: Saving the predicted segmentation masks to {out_dir}")

    # Define you own model
    model = build_model()
    model.to(device)

    # Define Loss function
    criterion = torch.nn.BCELoss()

    # Define Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Write the training loop!
    print("[INFO]: Starting training...")
    for epoch in range(num_epochs):
        model.train()
        print('****************************')
        print(epoch)
        print('****************************')

        for i, (image, gt_mask) in enumerate(train_dataloader):
            print('Minibatch ' + str(i))
            image = image.to(device)
            gt_mask = gt_mask.to(device)

            optimizer.zero_grad()

            # Forward pass
            output = model(image)
            output = torch.nn.Sigmoid()(output)
            loss = criterion(output, gt_mask.float())

            # Backward pass
            loss.backward()
            optimizer.step()

            # Trace output:
            print("         Training Loss: {}".format(loss.data.cpu().numpy()),
                  "- IoU: {}".format(compute_iou(output.data.cpu().numpy() > 0.5, gt_mask.data.cpu().numpy())))

        # Save model (after every checkpoint)
        torch.save(model.state_dict(), os.path.join(ckpt_dir, "last_epoch.pth"))

    # run on test data
    image_ids = []
    pred_masks = []

    with torch.no_grad():
        for i, (test_image, image_id) in enumerate(test_dataloader):
            test_image = test_image.to(device)
            image_id = image_id[0]  # unpack from batch (batch_size=1)

            # Forward pass
            test_output = model(test_image)
            test_output = torch.nn.Sigmoid()(test_output)

            # convert to binary image mask:
            pred_mask = (test_output > 0.5).squeeze().cpu().numpy()

            # Save the predicted mask as image (for your visualisation) - do not to submit these files!
            pred_mask_image = Image.fromarray(pred_mask)
            pred_mask_image.save((os.path.join(out_dir, image_id + "_mask.png")))

            # update lists of image ids and masks (will be used to later convert it into submission format):
            image_ids += [image_id]
            pred_masks += [pred_mask]

        # Kaggle submission format
        save_predictions(image_ids=image_ids, pred_masks=pred_masks, save_path=os.path.join(out_dir, 'submission.csv'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SML Project 2.")
    parser.add_argument(
        "-d",
        "--data_root",
        default="./datasets",
        help="Path to the datasets folder.",
    )
    parser.add_argument(
        "--ckpt_dir",
        default="./checkpoints",
        help="Path to save the model checkpoints to.",
    )
    args = parser.parse_args()

    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    ckpt_dir = os.path.join(args.ckpt_dir, dt_string)
    os.makedirs(ckpt_dir, exist_ok=True)
    print("[INFO]: Model checkpoints will be saved to:", ckpt_dir)

    # Set data root
    train_data_root = os.path.join(args.data_root, "train_data")
    print(f"[INFO]: Train data root: {train_data_root}")

    test_data_root = os.path.join(args.data_root, "test_data")
    print(f"[INFO]: Test data root: {test_data_root}")

    train(ckpt_dir, train_data_root, test_data_root)
