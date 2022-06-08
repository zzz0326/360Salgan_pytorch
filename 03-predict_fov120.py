import sys
from model import VGG16, bgr_mean
import torch
import os
import glob
import cv2
from tqdm import tqdm
import numpy as np


def test(path_to_images, path_output_maps, model_to_test=None):
    list_img_files = [k.split('\\')[-1].split('.')[0] for k in glob.glob(os.path.join(path_to_images, '*jpg'))]
    # Load Data
    list_img_files.sort()
    for curr_file in tqdm(list_img_files):
        test = cv2.imread(os.path.join(path_to_images, curr_file + '.jpg'), cv2.IMREAD_COLOR)
        image = cv2.resize(test, (256, 192), interpolation=cv2.INTER_AREA)

        image = np.transpose(image, [2, 0, 1])
        image = image.astype(np.float32)
        image -= bgr_mean[:, np.newaxis, np.newaxis]
        X = torch.from_numpy(image)
        X = X.unsqueeze(0)

        out = model_to_test(X).detach().numpy()
        out = out[0,]
        out = out.transpose(1, 2, 0)
        out = (out * 255).astype(np.uint8)

        out = cv2.resize(out, (test.shape[1], test.shape[0]), interpolation=cv2.INTER_CUBIC)
        out = cv2.GaussianBlur(out, (5, 5), 0)
        out = np.clip(out, 0, 255)

        cv2.imwrite(path_output_maps + curr_file + '.jpg', out)


def main(in_folder, out_folder):
    # Create network
    model = VGG16().cpu()
    # Here need to specify the epoch of model sanpshot
    weight = torch.load(
        "./120.pkl", map_location='cpu')

    model.load_state_dict(weight, strict=True)
    del weight
    #load_weights(model.net['output'], path= 'gen_',epochtoload=90, layernum=54)
    # Here need to specify the path to images and output path
    test(path_to_images=in_folder, path_output_maps=out_folder, model_to_test=model)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise NotImplementedError
    elif len(sys.argv) == 3:
        image_fol = sys.argv[1]
        print ('Image folder is %s' % image_fol)
        output_fol = sys.argv[2]
        print ('Saliency folder is %s' % output_fol)
        main(image_fol, output_fol)
    else:
        raise NotImplementedError