import gdown
import subprocess
import os
import requests   

if __name__ == "__main__":
    # Downloads Flickr8k
    if 'Images' not in os.listdir():
        url = 'https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip'
        output = 'Images.zip'
        r = requests.get(url)
        with open(output, 'wb') as f:
            f.write(r.content)
    
    # Download custom images
    if 'custom_images' not in os.listdir():
        os.mkdir('custom_images')  
        url = 'https://drive.google.com/drive/folders/13Ju7hVDg8mg-yyLDKNSk5NtL2vpXDHbj?usp=sharing'   
        output = 'custom_images'
        gdown.download_folder(url, quiet=False)  
        image_names = [file for file in os.listdir() if '.jpg' in file]
        for name in image_names:
            os.rename(name, './custom_images/' + name)
    else:
        print('File already exist ./custom_images')

    # download csv with captions
    if 'test_captions.csv' not in os.listdir() and 'train_captinos.csv' not in os.listdir():
        url = 'https://drive.google.com/drive/folders/1u_MLwdUCWmi1UHwTmpK7XLTykG2Ss_Ut?usp=sharing'
        gdown.download_folder(url, quiet=False)   
    else:
        print('File already exist: test_captions.csv, train_captions.csv')
    
    
    
    
 