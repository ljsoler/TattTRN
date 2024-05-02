import os
import shutil
import time
import cv2
import numpy as np


def add_bodypart_prefix_to_image_names(log_dir: str, data_path: str):
    print('[INFO] Adding bodypart prefix to image names')
    start = time.time()
    log = ''
    leg = data_path + 'NTU_Lower_Leg/'
    chest = data_path + 'NTU_Chest/'
    back = data_path + 'NTU_Back/'
    for filename in os.listdir(leg):
        os.rename(leg + filename, leg + 'leg_' + filename)
        log += filename + ' renamed to leg_' + filename + '\n'
    for filename in os.listdir(chest):
        os.rename(chest + filename, chest + 'chest_' + filename)
        log += filename + ' renamed to chest_' + filename + '\n'
    for filename in os.listdir(back):
        os.rename(back + filename, back + 'back_' + filename)
        log += filename + ' renamed to back_' + filename + '\n'

    log += '[SUMMARY]\n'
    log += 'Renamed ' + str(len(log.split('\n'))) + ' files.'
    with open(log_dir + 'add_bodypart_prefix_to_image_names.log', 'w+') as file:
        file.write(log)
    end = time.time()
    print('[INFO] Adding bodypart prefix to image names successful\n')
    print('[INFO] Time elapsed: ' + str(int((end - start) / 60)) + "m " + str(int((end - start) % 60)) + "s\n")


def rename_wrongly_named_images(log_dir: str, img_dir: str, base_img_dir: str):
    print('[INFO] Renaming files')
    start = time.time()
    log = ''
    file_names = []
    for filename in os.listdir(base_img_dir):
        if '_' in filename and (filename.endswith('.jpeg') or filename.endswith('.jpg') or filename.endswith('.png')):
            file_names.append(filename.split('.')[0])

    for file in os.listdir(img_dir):
        for filename in file_names:
            if filename in file:
                new_base_name = filename.replace('_', '-')
                new_image_name = new_base_name + file[len(filename):]
                os.rename(img_dir + file, img_dir + new_image_name)
                log += file + ' renamed to ' + new_image_name + '\n'

    log += '[SUMMARY]\n'
    log += 'Renamed ' + str(len(log.split('\n'))) + ' files.'

    # save log to file log_dir + 'rename_wrongly_named_images.log'
    with open(log_dir + 'rename_wrongly_named_images.log', 'w+') as file:
        file.write(log)

    end = time.time()
    print('[INFO] Renaming successful')
    print('[INFO] Time elapsed: ' + str(int((end - start) / 60)) + "m " + str(int((end - start) % 60)) + "s\n")


def filter_out_templates_without_images(log_dir: str, img_dir: str, templates_dir: str):
    print('[INFO] Filtering out templates without images')
    start = time.time()
    folder_log = ''
    template_log = ''

    folders = []
    # remove all empty folders from img_dir
    for folder in os.listdir(img_dir):
        if not os.listdir(img_dir + folder):
            os.rmdir(img_dir + folder)
            folder_log += 'Folder ' + folder + ' removed\n'
        else:
            folders.append(folder)

    # remove all templates from templates_dir that don't have corresponding folder in img_dir
    for template in os.listdir(templates_dir):
        template_name = template.split('.')[0]
        if template_name not in folders:
            # remove file template from templates_dir
            os.remove(templates_dir + template)
            template_log += 'Template ' + template + ' removed\n'

    log = folder_log + template_log
    log += '[SUMMARY]\n'
    log += 'Removed ' + str(len(folder_log.split('\n')) - 1) + ' folders.\n'
    log += 'Removed ' + str(len(template_log.split('\n')) - 1) + ' templates.'

    with open(log_dir + 'filter_out_templates_without_images.log', 'w+') as file:
        file.write(log)

    end = time.time()
    print('[INFO] Filtering successful')
    print('[INFO] Time elapsed: ' + str(int((end - start) / 60)) + "m " + str(int((end - start) % 60)) + "s\n")


def extract_bounding_box(image):
    nonzero_indices = np.nonzero(image)
    min_row = np.min(nonzero_indices[0])
    max_row = np.max(nonzero_indices[0])
    min_col = np.min(nonzero_indices[1])
    max_col = np.max(nonzero_indices[1])
    return min_row, max_row, min_col, max_col


def crop_images_by_pixel_mask(log_dir: str, img_dir: str, mask_dir: str, out_dir: str):
    print('[INFO] Cropping images by pixel mask')
    start = time.time()
    mask_log = ""
    tattoo_log = ""

    images_count = len(os.listdir(img_dir))
    for i, filename in enumerate(os.listdir(img_dir)):
        image = cv2.imread(os.path.join(img_dir, filename))
        mask = cv2.imread(os.path.join(mask_dir, filename))
        if mask is None:
            mask_log += "No mask for " + filename + "\n"
            continue

        # convert to grayscale
        mask = cv2.cvtColor(mask, cv2.COLOR_RGBA2GRAY)

        # segment
        # todo: maybe switch to use of histogram?
        #       hist = cv2.calcHist([mask], [0], None, [256], [0, 256])
        mask[mask > 2 / 3 * 255] = 0
        mask[mask < 1 / 3 * 255] = 0
        mask[mask != 0] = 255
        if np.count_nonzero(mask) == 0:
            tattoo_log += "No tattoo: " + filename + "\n"
            continue

        # convert back to BGR
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # apply mask
        segmented = cv2.bitwise_and(image, mask)

        # save segmented
        # cv2.imwrite(SEG_PATH + filename, segmented)

        # crop
        nonzero_indices = np.nonzero(segmented)
        min_row = np.min(nonzero_indices[0])
        max_row = np.max(nonzero_indices[0])
        min_col = np.min(nonzero_indices[1])
        max_col = np.max(nonzero_indices[1])
        cropped = segmented[min_row:max_row + 1, min_col:max_col + 1]
        # todo: is it necessary?
        #       set background to white
        cropped[cropped == 0] = 255

        # save cropped
        # overwrite original image
        template_name = '_'.join(filename.split('_')[1:-3])
        os.makedirs(os.path.join(out_dir, template_name), exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, template_name, filename), cropped)

        if (i + 1) % 50 == 0:
            print("Progress: " + str(i + 1) + "/" + str(images_count))

    log = mask_log + tattoo_log
    log += '[SUMMARY]\n'
    log += 'Cropped ' + str(i) + ' images.'

    with open(log_dir + 'crop_images.log', 'w+') as file:
        file.write(log)

    end = time.time()
    print('[INFO] Cropping successful')
    print('[INFO] Time elapsed: ' + str(int((end - start) / 60)) + "m " + str(int((end - start) % 60)) + "s\n")


def crop_images_by_bounding_box(log_dir: str, img_dir: str, mask_dir: str, out_dir: str, bb_ratio_threshold: float):
    print('[INFO] Cropping images by bounding box')
    start = time.time()
    mask_log = ""
    tattoo_log = ""
    bb_mask_log = ""
    images_count = len(os.listdir(img_dir))
    cropped_images_count = 0

    for i, filename in enumerate(os.listdir(img_dir)):
        image = cv2.imread(os.path.join(img_dir, filename))
        mask = cv2.imread(os.path.join(mask_dir, filename))
        if mask is None:
            mask_log += "No mask: " + filename + "\n"
            continue

        # convert to grayscale
        mask = cv2.cvtColor(mask, cv2.COLOR_RGBA2GRAY)

        bb_mask = mask.copy()
        bb_mask[bb_mask > 2 / 3 * 255] = 0
        bb_mask[bb_mask < 1 / 3 * 255] = 0
        bb_mask[bb_mask != 0] = 255
        if np.count_nonzero(bb_mask) == 0:
            tattoo_log += "No tattoo: " + filename + "\n"
            continue
        min_row, max_row, min_col, max_col = extract_bounding_box(bb_mask)
        bb_pixels = (max_row + 1 - min_row) * (max_col + 1 - min_col)

        # get skin count
        skin = mask.copy()
        skin[skin > 1 / 3 * 255] = 255
        skin[skin < 1 / 3 * 255] = 0
        skin_pixels = cv2.countNonZero(skin)

        bb_ratio = bb_pixels / skin_pixels
        if bb_ratio < bb_ratio_threshold:
            bb_mask_log += 'BB Ratio: {:.2f}'.format(bb_ratio) + " removing " + filename + "\n"
            continue

        cropped = image[min_row:max_row + 1, min_col:max_col + 1]
        template_name = '_'.join(filename.split('_')[1:-3])
        os.makedirs(os.path.join(out_dir, template_name), exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, template_name, filename), cropped)
        cropped_images_count += 1

        if (i + 1) % 50 == 0:
            print("Progress: " + str(i + 1) + "/" + str(images_count))


    log = '[MASKS LOG]\n' + mask_log + '[TATTOOS LOG]\n' + tattoo_log + '[BOUNDING BOX LOG]\n' + bb_mask_log
    log += '[SUMMARY]\n'
    log += 'Iterated through ' + str(i + 1) + ' images.\n'
    log += 'No mask for ' + str(len(mask_log.split('\n')) - 1) + ' images.\n'
    log += 'No tattoo for ' + str(len(tattoo_log.split('\n')) - 1) + ' images.\n'
    log += 'BB ratio too small for ' + str(len(bb_mask_log.split('\n')) - 1) + ' images.\n'
    log += 'BB ratio threshold: ' + str(bb_ratio_threshold) + '.\n'
    log += 'Cropped ' + str(cropped_images_count) + ' images.\n'

    with open(log_dir + 'crop_images_by_bounding_box.log', 'w+') as file:
        file.write(log)

    end = time.time()
    print('[INFO] Cropping successful')
    print('[INFO] Time elapsed: ' + str(int((end - start) / 60)) + "m " + str(int((end - start) % 60)) + "s\n")


def run(data_path: str):
    img_dir = data_path + 'images/'
    log_dir = data_path + 'logs/'
    base_img_dir = data_path + 'base_images/'
    templates_dir = data_path + 'tattoo_templates/'
    mask_dir = data_path + 'masks/'
    out_dir = data_path + 'tattoos/'
    bb_ratio_threshold = 0.15

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # add_bodypart_prefix_to_image_names(log_dir, data_path)
    rename_wrongly_named_images(log_dir, img_dir, base_img_dir)
    rename_wrongly_named_images(log_dir, mask_dir, base_img_dir)
    crop_images_by_bounding_box(log_dir, img_dir, mask_dir, out_dir, bb_ratio_threshold)
    filter_out_templates_without_images(log_dir, out_dir, templates_dir)


DATA_PATH = '/mnt/c/Users/Maciek/Desktop/DTU/semester_3/biometrics/tattoo-retrieval/data/work/'

run(DATA_PATH)