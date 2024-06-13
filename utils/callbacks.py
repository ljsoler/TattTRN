import pytorch_lightning as pl
import torch
import torchvision.utils
import os
import numpy as np
import torch.nn.functional as F
# from losses import ArcFace
from pyeer.eer_info import get_eer_stats
from utils.ploters.pyeer.eer_info import get_eer_stats
from pyeer.cmc_stats import load_scores_from_file, get_cmc_curve
import random
from numpy.linalg import norm
import matplotlib.pyplot as plt
from pathlib import Path

class ImagePredictionVAELogger(pl.Callback):
    def __init__(self, run_dir, data_module, num_samples):
        super().__init__()
        self.data_module = data_module
        self.num_samples = num_samples
        self.run_dir = run_dir

        # Fetch data from the validation set
        val_imgs = []
        val_templates = []

        # pick images to save predictions for validation step
        for batch in self.data_module.val_dataloader():
            val_imgs.append(batch[0])
            val_templates.append(batch[1])

        self.val_imgs = torch.cat(val_imgs)
        self.val_templates = torch.cat(val_templates)
        self.idxs = torch.randperm(len(self.val_imgs))[:self.num_samples]
    
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            print(f'\n[INFO] Saving validation images at sanity check')
            print(f'\n[INFO] IDs of images that will be logged: {self.idxs}')
            # Saving base images
            val_imgs = self.val_imgs[self.idxs].to(device=pl_module.device)
            # val_imgs = self.data_module.rollback_normalisation(val_imgs)
            img_grid = torchvision.utils.make_grid(val_imgs, nrow=self.num_samples)
            img_path = os.path.join(self.run_dir, f'val_imgs.jpg')
            torchvision.utils.save_image(img_grid, img_path)
            # Saving templates
            val_templates = self.val_templates[self.idxs].to(device=pl_module.device)
            template_grid = torchvision.utils.make_grid(val_templates, nrow=self.num_samples)
            template_path = os.path.join(self.run_dir, f'val_templates.jpg')
            torchvision.utils.save_image(template_grid, template_path)
        else:
            print(f'\n[INFO] Generating predictions at epoch {trainer.current_epoch}')
            val_imgs = self.val_imgs[self.idxs].to(device=pl_module.device)
            preds_t, preds_i = pl_module(val_imgs)
            temp_path = os.path.join(self.run_dir, f'epoch_{trainer.current_epoch}_template_predictions.jpg')
            img_path = os.path.join(self.run_dir, f'epoch_{trainer.current_epoch}_image_predictions.jpg')
            torchvision.utils.save_image(preds_t, temp_path,
                          nrow=self.num_samples)
            torchvision.utils.save_image(preds_i, img_path,
                          nrow=self.num_samples)
            
    def on_test_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            print(f'\n[INFO] Saving validation images at sanity check')
            print(f'\n[INFO] IDs of images that will be logged: {self.idxs}')
            # Saving base images
            val_imgs = self.val_imgs[self.idxs].to(device=pl_module.device)
            # val_imgs = self.data_module.rollback_normalisation(val_imgs)
            img_grid = torchvision.utils.make_grid(val_imgs, nrow=self.num_samples)
            img_path = os.path.join(self.run_dir, f'val_imgs.jpg')
            torchvision.utils.save_image(img_grid, img_path)
            # Saving templates
            val_templates = self.val_templates[self.idxs].to(device=pl_module.device)
            template_grid = torchvision.utils.make_grid(val_templates, nrow=self.num_samples)
            template_path = os.path.join(self.run_dir, f'val_templates.jpg')
            torchvision.utils.save_image(template_grid, template_path)
        else:
            print(f'\n[INFO] Generating predictions at epoch {trainer.current_epoch}')
            val_imgs = self.val_imgs[self.idxs].to(device=pl_module.device)
            preds_t, preds_i = pl_module(val_imgs)
            temp_path = os.path.join(self.run_dir, f'epoch_{trainer.current_epoch}_template_predictions.jpg')
            img_path = os.path.join(self.run_dir, f'epoch_{trainer.current_epoch}_image_predictions.jpg')
            torchvision.utils.save_image(preds_t, temp_path,
                          nrow=self.num_samples)
            torchvision.utils.save_image(preds_i, img_path,
                          nrow=self.num_samples)


class ImageReconstruction(pl.Callback):
    def __init__(self, output_dir, data_module):
        super().__init__()
        self.data_module = data_module
        self.output_dir = output_dir

    def on_test_epoch_end(self, trainer, pl_module):
        dataset = {}
        for i, batch in enumerate(self.data_module):
            x, l = batch
            x = x.to(device=pl_module.device)
            l = l.to(device=pl_module.device)
            feat_template, feat_image = pl_module(x)
            filenames = self.data_module.dataset.imgs[i]
            output_folder = os.path.join(self.output_dir, Path(filenames[0]).parent.name)
            os.makedirs(output_folder, exist_ok=True)
            output_folder_template = os.path.join(output_folder, '{}_temp.png'.format(Path(filenames[0]).stem))
            output_folder_image = os.path.join(output_folder, '{}_img.png'.format(Path(filenames[0]).stem))
            torchvision.utils.save_image(feat_template, output_folder_template)
            torchvision.utils.save_image(feat_image, output_folder_image)
            


class ImagePredictionLogger(pl.Callback):
    def __init__(self, run_dir, data_module, num_samples = 5):
        super().__init__()
        self.data_module = data_module
        self.num_samples = num_samples
        self.run_dir = run_dir

        # Fetch data from the validation set
        val_imgs = []
        val_templates = []

        # pick images to save predictions for validation step
        for batch in self.data_module.val_dataloader():
            val_imgs.append(batch[0])
            val_templates.append(batch[1])

        self.val_imgs = torch.cat(val_imgs)
        self.val_templates = torch.cat(val_templates)
        self.idxs = torch.randperm(len(self.val_imgs))[:self.num_samples]
    

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            print(f'\n[INFO] Saving validation images at sanity check')
            print(f'\n[INFO] IDs of images that will be logged: {self.idxs}')
            # Saving base images
            val_imgs = self.val_imgs[self.idxs].to(device=pl_module.device)
            val_imgs = self.data_module.rollback_normalisation(val_imgs)
            img_grid = torchvision.utils.make_grid(val_imgs, nrow=self.num_samples)
            img_path = os.path.join(self.run_dir, f'val_imgs.jpg')
            torchvision.utils.save_image(img_grid, img_path)
            # Saving templates
            val_templates = self.val_templates[self.idxs].to(device=pl_module.device)
            template_grid = torchvision.utils.make_grid(val_templates, nrow=self.num_samples)
            template_path = os.path.join(self.run_dir, f'val_templates.jpg')
            torchvision.utils.save_image(template_grid, template_path)
        else:
            print(f'\n[INFO] Generating predictions at epoch {trainer.current_epoch}')
            val_imgs = self.val_imgs[self.idxs].to(device=pl_module.device)
            template, img, _, _ = pl_module(val_imgs)
            preds_i = self.data_module.rollback_normalisation(img)
            preds_t = self.data_module.rollback_normalisation(template)
            temp_path = os.path.join(self.run_dir, f'epoch_{trainer.current_epoch}_template_predictions.jpg')
            img_path = os.path.join(self.run_dir, f'epoch_{trainer.current_epoch}_image_predictions.jpg')
            # result = self.plot_reconstructed(preds, n=3)
            # plt.imsave(img_path, result)
            #     plt.imsave(img_path, preds[idx], cmap='viridis')
            torchvision.utils.save_image(preds_t, temp_path,
                          nrow=self.num_samples)
            torchvision.utils.save_image(preds_i, img_path,
                          nrow=self.num_samples)
    

    def on_test_epoch_end(self, trainer, pl_module):
        dataset = {}
        for i, batch in enumerate(self.data_module):
            x, l = batch
            x = x.to(device=pl_module.device)
            l = l.to(device=pl_module.device)
            feat_template, feat_image = pl_module(x)
            filenames = self.data_module.imgs[i]
            features = torch.cat((feat_image, feat_template), dim=1)
            features = F.normalize(features)
            features = features.detach().cpu().numpy()
            label = l.detach().cpu().numpy()
            for i in range(label.shape[0]):
                if label[i] in dataset:
                    dataset[label[i]].append(features[i])
                else:
                    dataset[label[i]] = [features[i]]
        
        gen, imp = self.perform_verification(dataset)
        stat = get_eer_stats(gen, imp)
        eer = torch.tensor(stat.eer*100, dtype=torch.float)
        fmr10 = torch.tensor(stat.fmr10*100, dtype=torch.float)
        fmr20 = torch.tensor(stat.fmr20*100, dtype=torch.float)
        fmr100 = torch.tensor(stat.fmr100*100, dtype=torch.float)
        pl_module.log_dict({'EER_Ver': eer, 'FMR10': fmr10, 'FMR20': fmr20, 'FMR100': fmr100})
    # def plot_reconstructed(self, x_hat, r0=(-5, 10), r1=(-10, 5), n=12):
    #     w = 224
    #     x_hat = x_hat.to('cpu').detach().numpy()
    #     img = np.zeros((n*w, n*w))
    #     for i, y in enumerate(np.linspace(*r1, n)):
    #         for j, x in enumerate(np.linspace(*r0, n)):
    #             img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = x_hat[i*n + j]
    #     return img
    #     # plt.imshow(img, extent=[*r0, *r1])
    


class CallBackVerification(pl.Callback):
    def __init__(self, data_module):
        super().__init__()
        self.data_module = data_module


    def on_validation_epoch_end(self, trainer, pl_module):
        dataset = {}
        for batch in self.data_module.val_dataloader():
            x, _, l = batch
            x = x.to(device=pl_module.device)
            l = l.to(device=pl_module.device)
            _, _, feat_template, feat_image = pl_module(x)
            features = torch.cat((feat_image, feat_template), dim=1)
            features = F.normalize(features)
            features = features.detach().cpu().numpy()
            label = l.detach().cpu().numpy()
            for i in range(label.shape[0]):
                if label[i] in dataset:
                    dataset[label[i]].append(features[i])
                else:
                    dataset[label[i]] = [features[i]]
        
        gen, imp = self.perform_verification(dataset)
        stat = get_eer_stats(gen, imp)
        eer = torch.tensor(stat.eer*100, dtype=torch.float)
        fmr10 = torch.tensor(stat.fmr10*100, dtype=torch.float)
        fmr20 = torch.tensor(stat.fmr20*100, dtype=torch.float)
        fmr100 = torch.tensor(stat.fmr100*100, dtype=torch.float)
        pl_module.log_dict({'eer': eer, 'fmr20': fmr20, 'fmr10': fmr10, 'fmr100': fmr100})


    def on_test_epoch_end(self, trainer, pl_module):
        dataset = {}
        for batch in self.data_module:
            x, l = batch
            x = x.to(device=pl_module.device)
            l = l.to(device=pl_module.device)
            _, _, feat_template, feat_image = pl_module(x)
            features = torch.cat((feat_image, feat_template), dim=1)
            features = F.normalize(features)
            features = features.detach().cpu().numpy()
            label = l.detach().cpu().numpy()
            for i in range(label.shape[0]):
                if label[i] in dataset:
                    dataset[label[i]].append(features[i])
                else:
                    dataset[label[i]] = [features[i]]
        
        gen, imp = self.perform_verification(dataset)
        stat = get_eer_stats(gen, imp)
        eer = torch.tensor(stat.eer*100, dtype=torch.float)
        fmr10 = torch.tensor(stat.fmr10*100, dtype=torch.float)
        fmr20 = torch.tensor(stat.fmr20*100, dtype=torch.float)
        fmr100 = torch.tensor(stat.fmr100*100, dtype=torch.float)

        del dataset
        pl_module.log_dict({'EER_VER': eer, 'FMR20': fmr20, 'FMR10': fmr10, 'FMR100': fmr100})

    #verification performance
    @staticmethod
    def perform_verification(dataset):

        keys = list(dataset.keys())
        random.shuffle(keys)
        genuine_list, impostors_list = [], []
        #Compute mated comparisons
        for k in keys:
            for i in range(len(dataset[k]) - 1):
                reference = dataset[k][i]
                for j in range(i + 1, len(dataset[k])):
                    probe = dataset[k][j]
                    value = np.dot(probe,reference)/(norm(probe)*norm(reference))
                    genuine_list.append(value)

        #Compute non-mated comparisons
        for i in range(len(keys)):
            reference = random.choice(dataset[keys[i]])
            for j in range(len(keys)):
                if i != j:
                    probe = random.choice(dataset[keys[j]])
                    value = np.dot(probe,reference)/(norm(probe)*norm(reference))
                    impostors_list.append(value)

        return genuine_list, impostors_list


import json
class CallBackGetSimilars(pl.Callback):
    def __init__(self, output_dir, data_module, top_k = 10):
        super().__init__()
        self.data_module = data_module
        self.top_k = top_k
        self.output_dir = output_dir

    def on_test_epoch_end(self, trainer, pl_module):
        dataset = []
        for i, batch in enumerate(self.data_module):
            x, l = batch
            x = x.to(device=pl_module.device)
            l = l.to(device=pl_module.device)
            _, _, feat_template, feat_image = pl_module(x)
            features = torch.cat((feat_image, feat_template), dim=1)
            features = F.normalize(features)
            features = features.detach().cpu().numpy()
            label = l.detach().cpu().numpy()
            filename = self.data_module.dataset.imgs[i][0]
            f_name = '{}/{}'.format(Path(filename).parent.name, Path(filename).name)
            dataset.append((f_name, features[0]))
        
        dictionary = self.get_similars(dataset, self.top_k)

        with open(os.path.join(self.output_dir, 'most_similar.json'), 'w') as fp:
            json.dump(dictionary, fp, sort_keys=True, indent=4)
        

    #verification performance
    @staticmethod
    def get_similars(dataset, top_k):
        dictionary = {}
        for i in range(len(dataset)): 
            l_r, f_r = dataset[i]

            temp_list = []
            for j in range(i, len(dataset)): 
                l_t, f_t = dataset[j]
                value = np.dot(f_t,f_r)/(norm(f_t)*norm(f_r))
                temp_list.append((l_t, str(value)))
            
            temp_list.sort(key=lambda v: float(v[1]), reverse=True)

            dictionary[l_r] = temp_list[ :top_k] if len(temp_list) >= top_k else temp_list
            
        return dictionary
        
    

class CallBackOpenSetIdentification(pl.Callback):
    def __init__(self, data_module, scores_dir: str):
        super().__init__()
        self.data_module = data_module
        self.scores_dir = scores_dir

    def on_test_epoch_end(self, trainer, pl_module):
        dataset = {}
        for batch in self.data_module:
            x, l = batch
            x = x.to(device=pl_module.device)
            l = l.to(device=pl_module.device)
            _, _, feat_template, feat_image = pl_module(x)
            features = torch.cat((feat_image, feat_template), dim=1)
            features = F.normalize(features)
            features = features.detach().cpu().numpy()
            label = l.detach().cpu().numpy()
            for i in range(label.shape[0]):
                if label[i] in dataset:
                    dataset[label[i]].append(features[i])
                else:
                    dataset[label[i]] = [features[i]]

        gen, imp = self.perform_identification(dataset)

        stat = get_eer_stats(gen, imp)
        eer = torch.tensor(stat.eer * 100, dtype=torch.float)
        fmr10 = torch.tensor(stat.fmr10 * 100, dtype=torch.float)
        fmr20 = torch.tensor(stat.fmr20 * 100, dtype=torch.float)
        fmr100 = torch.tensor(stat.fmr100 * 100, dtype=torch.float)
        pl_module.log_dict({'EER_Ident': eer, 'FNIR10': fmr10, 'FNIR20': fmr20, 'FNIR100': fmr100})

        # save the results to a file
        score_path = os.path.join(self.scores_dir, 'open_set_scores.npz')
        np.savez(score_path, gen=gen, imp=imp)

    #verification performance
    @staticmethod
    def perform_identification(dataset, k_fold=10):
        keys = list(dataset.keys())
        random.shuffle(keys)
        k = int(len(keys)/k_fold)
        total_index = list(np.arange(len(keys)))
        genuine_list, impostors_list = [], []

        for i in range(k_fold):
            start = k*i
            end = k*(i + 1)
            impostors_id = {k:dataset[k] for k in keys[start:end]}
            var_iter = set(np.arange(start, end))
            gen_idx = list(set(total_index) - var_iter)
            gen_keys = [keys[k] for k in gen_idx]
            genuines_id = {k:dataset[k] for k in gen_keys}
            enrolled_database, search_gen_database = [], []

            for g in genuines_id:
                aux = random.sample(genuines_id[g], 2)
                enrolled_database.append(aux[0])
                search_gen_database.append(aux[1])

            #Compute mated comparisons
            for s in search_gen_database:
                probe = s
                temp_list = []
                for r in enrolled_database:
                    reference = r
                    value = np.dot(probe,reference)/(norm(probe)*norm(reference))
                    temp_list.append(value)
                temp_list.sort(reverse=True)
                genuine_list.append(temp_list[0])

            #Compute non-mated comparisons
            for nm in impostors_id:
                for v in impostors_id[nm]:
                    probe = v
                    temp_list = []
                    for r in enrolled_database:
                        reference = r
                        value = np.dot(probe,reference)/(norm(probe)*norm(reference))
                        temp_list.append(value)
                    temp_list.sort(reverse=True)
                    impostors_list.append(temp_list[0])

        return genuine_list, impostors_list


class CallbackCloseSetIdentification(pl.Callback):
    def __init__(self, data_module, scores_dir: str, rank_value = 20):
        super().__init__()
        self.data_module = data_module
        self.scores_dir = scores_dir
        self.rank_value = rank_value


    def on_test_epoch_end(self, trainer, pl_module):
        dataset = {}
        for batch in self.data_module:
            x, l = batch
            x = x.to(device=pl_module.device)
            l = l.to(device=pl_module.device)
            _, _, feat_template, feat_image = pl_module(x)
            features = torch.cat((feat_image, feat_template), dim=1)
            features = F.normalize(features)
            features = features.detach().cpu().numpy()
            label = l.detach().cpu().numpy()
            for i in range(label.shape[0]):
                if label[i] in dataset:
                    dataset[label[i]].append(features[i])
                else:
                    dataset[label[i]] = [features[i]]

        final_scores, final_mated_comparisons = self.perform_identification(dataset)

        ranks = []

        # save final scores to txt file
        for fold in range(len(final_scores)):
            score_path = os.path.join(self.scores_dir, 'close_set_scores_{}.txt'.format(fold))
            score_tp_path = os.path.join(self.scores_dir, 'close_set_scores_tp_{}.txt'.format(fold))
            np.savetxt(score_path, final_scores[fold])
            np.savetxt(score_tp_path, final_mated_comparisons[fold])
            print('[INFO] Saved closed set scores to {}'.format(score_path))
            print('[INFO] Saved closed set scores true positive to {}'.format(score_tp_path))
            #Computing rank-1
            scores_comb = load_scores_from_file(score_path, score_tp_path)
            ranks.append(get_cmc_curve(scores_comb, self.rank_value))
        
        mean_ranks = np.array(ranks).mean(axis=0)
        # std_ranks = np.array(ranks).std(axis=1)
        pl_module.log_dict({'Rank-1': mean_ranks[0]*100, 'Rank-10': mean_ranks[9]*100, 'Rank-20': mean_ranks[19]*100})

    def perform_identification(self, dataset, k_fold=10):
        keys = list(dataset.keys())
        final_scores, final_mated_comparisons = [], []

        for i in range(k_fold):
            random.shuffle(keys)
            mated_comparisons, scores = [], []

            enrolled_database, search_gen_database = [], []
            # creating enrollment and query databases
            for k in keys:
                aux = random.sample(dataset[k], 2)
                enrolled_database.append((k, aux[0]))
                search_gen_database.append((k, aux[1]))

            # Compute mated comparisons
            for k1, s in search_gen_database:
                probe = s

                for k2, r in enrolled_database:
                    reference = r
                    value = np.dot(probe, reference) / (norm(probe) * norm(reference))
                    scores.append((k1, k2, value))

                    if k1 == k2:
                        mated_comparisons.append((k1, k2))

            final_scores.append(scores.copy())
            final_mated_comparisons.append(mated_comparisons.copy())

        return final_scores, final_mated_comparisons