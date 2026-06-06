
import os
import numpy as np

from pathlib import Path

from PIL import Image



def preprocess():

    # read raw data
    raw_data_path = Path('dataset/raw')
    dataset_path = Path('dataset/proc')

    for folder in raw_data_path.iterdir():

        partition_name = folder.name

        images = folder / 'images'
        labels = folder / 'labels'

        dataset_image_path = dataset_path / partition_name
        dataset_image_path.mkdir(parents=True, exist_ok=True)

        list_name = partition_name + '.txt'

        name_list = []
        label_list = []
        
        for item in sorted(labels.iterdir()):

            file_name = item.stem
            image = images / (file_name + '.jpg') 

            img = Image.open(image)
            img = np.asarray(img)

            index = 0

            with open(item, 'r') as file:
                for line in file:
                    
                    line_items = line.split(' ')
                    cls = line_items[0]
                    x_center = float(line_items[1]) * img.shape[1]
                    y_center = float(line_items[2]) * img.shape[0]
                    width = float(line_items[3]) * img.shape[1]
                    height = float(line_items[4]) * img.shape[0]


                    element = img[int(y_center - 0.5 * height):int(y_center + 0.5 * height), int(x_center - 0.5 * width):int(x_center + 0.5 * width), :]
                    

                    out = Image.fromarray(element)
                    out.save(dataset_image_path / (file_name + '_' + str(index) + '.jpg'))

                    name_list.append((file_name + '_' + str(index) + '.jpg'))
                    label_list.append(cls)
                    
                    index += 1

        with open(dataset_image_path / list_name, 'w') as file:
            
            for name, cls in zip(name_list, label_list):
                file.write(f"{name} {cls}\n")
            
            
        
        

    
    



