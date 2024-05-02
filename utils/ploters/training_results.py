import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import argparse


'''
Prerequisites:
    - copy metrics.csv from lightning_logs to run folder
    - install matplotlib-3.8.2
'''


def create_comparison_image(run_dir, result_dir, images_count, take_first = False):
    # Load validation images and templates
    val_imgs = Image.open(os.path.join(run_dir, 'val_imgs.jpg'))
    val_templates = Image.open(os.path.join(run_dir, 'val_templates.jpg'))

    all_images = [int(name.split('_')[1]) for name in os.listdir(run_dir) if name.startswith('epoch_')]
    all_images.sort()
    all_images = [f'epoch_{i}_predictions.jpg' for i in all_images]
    epoch_count = len(all_images)
    epoch_images = []
    epoch_images_names = []

    # Calculate the step size for epoch images
    if take_first:
        step_size = 1
    else:
        step_size = max(1, (epoch_count - 1) // (images_count - 1))

    for i in range(0, epoch_count, step_size):
        img_path = os.path.join(run_dir, all_images[i])
        epoch_images.append(Image.open(img_path))
        epoch_images_names.append(all_images[i])

    # If more images are loaded, remove extra images
    if len(epoch_images) > images_count:
        epoch_images = epoch_images[:images_count]
        epoch_images_names = epoch_images_names[:images_count]

    labels = ['Cropped image'] + [str(img) for img in epoch_images_names] + ['Template']

    # Combine all images
    images = [val_imgs] + epoch_images + [val_templates]

    # Create a new image with enough space for all images stacked vertically
    new_img = Image.new('RGB', (images[0].width, images[0].height * len(images)))

    # Paste each image into the new image
    for i, img in enumerate(images):
        new_img.paste(img, (0, i * img.height))

    # Create a new figure and show the image
    plt.figure(figsize=(10, 20))
    plt.imshow(np.array(new_img))
    plt.yticks([img.height / 2 + img.height * i for i in range(len(images))], labels)
    plt.tick_params(left=False, bottom=False, labelbottom=False)
    plt.box(False)

    # Save the figure
    if take_first:
        plt.savefig(os.path.join(result_dir, f'comparison_first_{images_count}_epochs.jpg'), bbox_inches='tight')
    else:
        plt.savefig(os.path.join(result_dir, f'comparison_{images_count}_epochs.jpg'), bbox_inches='tight')


def create_loss_graph(run_dir, result_dir):
    # Load the metrics CSV file
    df = pd.read_csv(os.path.join(run_dir, 'metrics.csv'))

    # Create a new figure
    plt.figure()

    template_df = df[df['template_loss'].notna()]
    train_df = df[df['train_loss'].notna()]
    feature_df = df[df['features_loss'].notna()]

    # Plot the training loss
    plt.plot(train_df['epoch'], train_df['train_loss'], label='Train Loss')
    plt.plot(template_df['epoch'], template_df['template_loss'], label='Template Loss')
    plt.plot(feature_df['epoch'], feature_df['features_loss'], label='Features Loss')

    # Add labels and legend
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Save the figure
    plt.savefig(os.path.join(result_dir, 'loss_graph.jpg'))

    # Create a new figure
    plt.figure()

    fmr20_df = df[df['fmr20'].notna()]
    fmr100_df = df[df['fmr100'].notna()]
    eer_df = df[df['eer'].notna()]
    fmr10_df = df[df['fmr10'].notna()]
    reconstruction_df = df[df['reconstruction_loss'].notna()]

    # Plot the training loss
    plt.plot(fmr10_df['epoch'], fmr10_df['fmr10'], label='FMR10')
    plt.plot(fmr20_df['epoch'], fmr20_df['fmr20'], label='FMR20')
    plt.plot(fmr100_df['epoch'], fmr100_df['fmr100'], label='FMR100')
    plt.plot(eer_df['epoch'], eer_df['eer'], label='EER')
    plt.plot(reconstruction_df['epoch'], reconstruction_df['reconstruction_loss'], label='Reconstruction Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()

    plt.savefig(os.path.join(result_dir, 'error_graph.jpg'))


def main(args):
    run_dir = os.path.join(args.root_dir, args.runs_dir, args.run_id)
    result_dir = os.path.join(run_dir, 'results')
    os.makedirs(os.path.join(run_dir, 'results'), exist_ok=True)

    create_comparison_image(run_dir, result_dir, 2, take_first=True)
    create_loss_graph(run_dir, result_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for generating training results')
    parser.add_argument('--root_dir', type=str, help="root dir")
    parser.add_argument('--runs_dir', type=str, help="name of the dir where runs are stored")
    parser.add_argument('--run_id', type=str, help="id of the run you want to generate results for")

    args_ = parser.parse_args()
    print(args_)
    main(args_)