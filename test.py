from torchvision.datasets import ImageFolder
from utils import TTN_wrapper
from utils import CallBackOpenSetIdentification, CallbackCloseSetIdentification, CallBackVerification
from datetime import datetime
import os
import pytorch_lightning as pl
import time
import argparse
from torchvision import transforms
import torch
import glob
import csv


def main(args):
    print('[INFO] Preparing directories')
    run_id = datetime.now().strftime('%m%d%H%M')
    run_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    scores_dir = os.path.join(run_dir, 'scores')
    os.makedirs(scores_dir, exist_ok=True)
    ckpt_path_folder = args.checkpoint_folder

    ckpt_path = glob.glob(os.path.join(ckpt_path_folder,  "epoch=*.ckpt"))[0]

    test_transform = transforms.Compose([
                    transforms.Resize((args.input_size, args.input_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    print('[INFO] Configuring data module')
    data_test = ImageFolder(
        root=args.images_dir,
        transform=test_transform
    )

    test_dataloader = torch.utils.data.DataLoader(data_test,
                                          batch_size=args.batch_size,
                                          shuffle=False,
                                          num_workers=args.num_workers
                                                  )

    print("Len of data_test: " + str(len(data_test.classes)))
    print("Len of test_dataloader: " + str(len(test_dataloader)))

    # model = TTN_wrapper.load_from_checkpoint('checkpoints/try_ckpt_epoch_1.ckpt')

    model = TTN_wrapper(model=args.backbone, num_features = args.num_features, num_identities=args.train_num_identities)

    # Initialise testing callbacks
    callbackVerification = CallBackVerification(test_dataloader)
    callbackOpenSetIdentification = CallBackOpenSetIdentification(test_dataloader, scores_dir)
    callbackCloseSetIdentification = CallbackCloseSetIdentification(test_dataloader, scores_dir)

    print('[INFO] Testing model')
    start_time = time.time()
    trainer = pl.Trainer(callbacks=[
        callbackVerification,
        callbackOpenSetIdentification,
        callbackCloseSetIdentification
    ])
    results = trainer.test(model, dataloaders=test_dataloader, ckpt_path=ckpt_path)[0]

    #Adding testing paramters
    results['Backbone'] = args.backbone
    results['Emb_size'] = args.num_features
    results['Margin'] = args.M

    exist = False
    if os.path.exists(args.csv_file):
        exist = True

    with open(args.csv_file, mode='a') as csv_file:
        fieldnames = ['Backbone', 'Emb_size', 'Margin', 'EER_Ver', 'FMR10', 'FMR20', 'FMR100', 'EER_Ident', 'FNIR10', 'FNIR20', 'FNIR100', 'Rank-1', 'Rank-10', 'Rank-20']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not exist:
            writer.writeheader()
        writer.writerow(results)

    print('The results are {}'.format(results))
    end_time = time.time()
    training_time = time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))

    #WRITE THE RESULTS IN THE OUTPUT FILE
    with open(os.path.join(run_dir, f'run_{run_id}.log'), 'w') as f:
        f.write(f'[RUN INFO]\n')
        f.write(f'Run ID: {run_id}\n')
        f.write(f'Testing time: {training_time}\n')
        f.write(f'\n')
        f.write(f'[DATA INFO]\n')
        f.write(f'Data folder: {args.images_dir}\n')
        f.write(f'\n')
        f.write(f'[MODEL SETTINGS]\n')
        f.write(f'Image size: {args.input_size}x{args.input_size}\n')
        f.write(f'backbone: {args.backbone}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch tattoo transformer network (TTN) for tattoo retrieval')
    parser.add_argument('--output_dir', type=str, help="output dir")
    parser.add_argument('--images_dir', type=str, help="image dir")
    parser.add_argument('--checkpoint_folder', type=str, help="checkpoint folder")
    parser.add_argument('--csv_file', type=str, help="path to the csv file where the results will be stored")
    parser.add_argument('--num_features', type=int, default=512, help="embedding size")
    parser.add_argument('--M', type=float, default=0.1, help="margin used by the Angular Margin")
    parser.add_argument('--input_size', type=int, default=224, help="image size to resize")
    parser.add_argument('--batch_size', type=int, default=32, help="batch size")
    parser.add_argument('--num_workers', type=int, default=2, help="number of workers to load the database")
    parser.add_argument('--train_num_identities', type=int, default=457, help="number of identities in train dataset")
    parser.add_argument('--backbone', type=str, default='mobilenet_v3_large',
                        choices=['mobilenet_v3_large', 'resnet101', 'densenet121', 'efficientnet_v2_s', 'swin_s'],
                        help="gesture to train or to evaluate")


    args_ = parser.parse_args()
    print(args_)
    main(args_)