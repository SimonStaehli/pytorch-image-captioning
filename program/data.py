import sys
import os
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import random
from multiprocess import Pool

def show_random_images(image_dir, captions_path, count=10, ncols=3):
    # load captions
    captions = pd.read_table(captions_path, sep=',')
    
    nrows =  int(count/ncols)+1
    fig = plt.subplots(figsize=(18, nrows*4))
    
    for i in range(1, count+1):
        plt.subplot(nrows, ncols, i)
        image_dirs = captions['image'].unique()
        random_image = random.choice(image_dirs)
        img = PIL.Image.open(f'{image_dir}/{random_image}')
        
        image_captions = captions.loc[captions['image'] == random_image, 'caption'].to_list()
        image_captions = [str(i+1)+ '. '+ cap for i, cap in enumerate(image_captions)]
        image_captions = '\n'.join(image_captions)
        plt.title(f'{image_captions}', loc='center', fontsize=8)
        plt.axis('off')
        plt.imshow(img)
    
    plt.subplots_adjust(hspace=.5)
    plt.show()


def copy_all_files(fp_from_to: list, n_proc=4):
    """Wrapper Function for Multiprocessing"""
    # Copy all Files
    with Pool(processes=n_proc) as pool:
        for _ in tqdm(pool.imap_unordered(_copy_to, fp_from_to), total=len(fp_from_to)):
            pass

def _copy_to(files_string):
    """Function for Multiprocessing"""
    from shutil import copyfile ## Questionable workaround??
    file_from, file_to = files_string.split('__')
    _ = copyfile(file_from, file_to)
    