#### This is the official repository of the paper:
### TattTRN: Template Reconstruction Network for Tattoo Retrieval
### Paper accepted at [CVPR 2024 Biometrics Workshop](https://www.vislab.ucr.edu/Biometrics2024/index.php)

<img src="utils/workflow.png"> 

<hr/>

## Requierements ##
- Python 3.8+
- pytorch-lightning==2.1.0
- torch==2.1.0
- pyeer

<hr/>

## TattTRN training ##
1. Ask for the semi-synthetic database for tattoo retrieval to lazaro-janier.gonzalez-soler@h-da.de or christian.rathgeb@h-da.de under the subject HDA-Tattoo-Retrieval.
2. run: python train.py --root_path . --data_dir path-to-synthetic_data --images_dir Images --templates_dir Tattoo_templates --runs_dir runs_dual --num_workers 7 --max_epochs 100 --val_split 0.2 --M 0.1 --L 4 --backbone efficientnet_v2_s --num_features 512 --val_freq 10 --batch_size 64

<hr/>

## TattTRN testing ##
1. Download Webtattoo or BIVTatt.
2. Organise the database in the following format:

```
.
├── train                   # Training set folder
│   ├── Tattoo-ID-1         # Tattoo ID category
│       ├── images1.jpg     # images
│       ├── images1.jpg             
│       └── ...             # etc.
│   ├── Tattoo-ID-2         # More tattoo categories
│       └── ...             # etc.
├── test                    # Test set folder similar to the training folder
│   └── ...                 # etc.     

```  

2. run: python test.py --images_dir image-folder --output_dir output-folder --checkpoint_folder path-to-checkpoint --backbone efficientnet_v2_s --csv_file path-to-csv --num_features 512 --M 0.1

<hr/>

## Pre-trained Models ##

Pre-trained models for the best parameter settings can be found in [Google Drive](https://drive.google.com/drive/folders/1BeaHueUogVKZ_G6Ja2kE8zHF3xYzDDiT?usp=sharing) 

<hr/>

## Citation ##
If you use any of the code provided in this repository or the models provided, please cite the following paper:
```
@InProceedings{GonzalezSoler-TattTRN-CVPRW-2024,
    author    = {L. Gonzalez-Soler and M. Salwowski and C. Rathgeb and Daniel Fischer},
    title     = {TattTRN: Template Reconstruction Network for Tattoo Retrieval},
    booktitle = {Proc. IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {}
}
```