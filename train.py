from utils import SyntheticImagesDataModule
from utils import TTN_wrapper
from utils import CallBackVerification, ImagePredictionLogger
from datetime import datetime
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import time
import argparse
from lightning.pytorch.loggers import CSVLogger


def main(args):
    print('[INFO] Preparing directories')
    data_path = os.path.join(args.root_path, args.data_dir)
    run_id = datetime.now().strftime('%m%d%H%M')
    run_dir = os.path.join(args.root_path, args.runs_dir, '{}_{}_{}_{}'.format(args.backbone, args.L, args.M, args.num_features))
    # os.makedirs(os.path.join(args.root_path, args.runs_dir), exist_ok=True)
    # os.makedirs(run_dir, exist_ok=True)

    print('[INFO] Configuring data module')
    data_module = SyntheticImagesDataModule(
        data_path=data_path,
        images_dir=args.images_dir,
        templates_dir=args.templates_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        input_size=args.input_size,
        use_grayscale=args.grayscale
    )

    data_module.setup()
    train_size = data_module.get_train_size()
    print(f'[INFO] Train size: {train_size}')
    train_templates_count = data_module.get_train_templates_count()
    print(f'[INFO] Train templates count: {train_templates_count}')
    val_size = data_module.get_val_size()
    print(f'[INFO] Validation size: {val_size}')
    val_templates_count = data_module.get_val_templates_count()
    print(f'[INFO] Validation templates count: {val_templates_count}')

    print('[INFO] Configuring model')
    model = TTN_wrapper(model=args.backbone, num_features = args.num_features, num_identities=train_templates_count, s = args.S, m = args.M, L=args.L)

    # Initialise callbacks
    callbackVerification = CallBackVerification(data_module)
    monitor = 'eer'
    filename = '{epoch}-{eer:.2f}'
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        dirpath=run_dir,
        filename=filename
    )

    logger = CSVLogger(run_dir, name="metrics")

    prediction_logger = ImagePredictionLogger(run_dir, data_module, args.epoch_grid_size)
    callbacks = [prediction_logger, callbackVerification, checkpoint_callback]


    print('[INFO] Training model')
    start_time = time.time()
    trainer = pl.Trainer(max_epochs=args.max_epochs, 
                         check_val_every_n_epoch = args.val_freq, 
                         logger=logger,
                         callbacks=callbacks, 
                         log_every_n_steps=1)
    trainer.fit(model, data_module)
    end_time = time.time()
    training_time = time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))

    print('[INFO] Saving model')
    trainer.save_checkpoint(os.path.join(run_dir, f'model_{run_id}.ckpt'))

    with open(os.path.join(run_dir, f'run_{run_id}.log'), 'w') as f:
        f.write(f'[RUN INFO]\n')
        f.write(f'Run ID: {run_id}\n')
        f.write(f'Training time: {training_time}\n')
        f.write(f'Number of epochs: {args.max_epochs}\n')
        f.write(f'\n')
        f.write(f'[DATA INFO]\n')
        f.write(f'Data folder: {args.data_dir}\n')
        f.write(f'Train size: {train_size}\n')
        f.write(f'Train templates count: {train_templates_count}\n')
        f.write(f'Validation size: {val_size}\n')
        f.write(f'Validation templates count: {val_templates_count}\n')
        f.write(f'\n')
        f.write(f'[MODEL SETTINGS]\n')
        f.write(f'Image size: {args.input_size}x{args.input_size}\n')
        f.write(f'Batch size: {args.batch_size}\n')
        f.write(f'Number of workers: {args.num_workers}\n')
        f.write(f'Validation split: {args.val_split}\n')
        f.write(f'Lambda: {args.L}\n')
        f.write(f'S: {args.S}\n')
        f.write(f'M: {args.M}\n')
        f.write(f'backbone: {args.backbone}\n')
        f.write(f'Grayscale: {args.grayscale}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch tattoo transformer network (TTN) for tattoo retrieval')
    parser.add_argument('--root_path', type=str, help="path to database")
    parser.add_argument('--data_dir', type=str, help="data dir")
    parser.add_argument('--images_dir', type=str, help="image dir")
    parser.add_argument('--templates_dir', type=str, help="template dir")
    parser.add_argument('--runs_dir', type=str, help="runs dir")
    parser.add_argument('--val_freq', type=int, default=10, help="frequency of validation")
    parser.add_argument('--num_features', type=int, default=512, help="embedding size")
    parser.add_argument('--epoch_grid_size', type=int, default=10, help="size of validation grid")
    parser.add_argument('--max_epochs', type=int, default=100, help="max number of epochs")
    parser.add_argument('--input_size', type=int, default=224, help="image size to resize")
    parser.add_argument('--batch_size', type=int, default=128, help="batch size")
    parser.add_argument('--num_workers', type=int, default=2, help="number of workers to load the database")
    parser.add_argument('--val_split', type=float, default=0.05, help="rate to build validation database")
    parser.add_argument('--L', type=float, default=0.5, help="Lambda value for loss function")
    parser.add_argument('--S', type=int, default=64, help="parameter used by the Angular Margin")
    parser.add_argument('--M', type=float, default=0.5, help="margin used by the Angular Margin")
    parser.add_argument('--grayscale', action='store_true', default=False,
                    help='convert the template to grayscale')
    parser.add_argument('--backbone', type=str, default='mobilenet_v3_large',
                        choices=['mobilenet_v3_large', 'resnet101', 'densenet121', 'efficientnet_v2_s', 'swin_s'],
                        help="gesture to train or to evaluate")

    args_ = parser.parse_args()
    print(args_)
    main(args_)