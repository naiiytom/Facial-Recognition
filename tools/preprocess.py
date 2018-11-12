
import os
import numpy as np
from scipy import misc
import util

class preprocess():
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir

    def collect_data(self):
        output_dir = os.path.expanduser(self.output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        dataset = util.get_dataset(self.input_dir)
        
        minsize= 20
        threshold = [0.6, 0.7, 0.7]
        factor = 0.709
        margin = 44
        image_size = 182

        random_key = np.random.randint(0, high=99999)
        bound_filename = os.path.join(output_dir, 'boundign_%05d.txt' % random_key)

        with open(bound_filename, 'w') as txt_file:
            nrof_image_total = 0
            nrof_successfully_aligned = 0
            for cls in dataset:
                output_class_dir = os.path.join(output_dir, cls.name)
                if not os.path.exists(output_class_dir):
                    os.makedirs(output_class_dir)
                for image_path im cls.image_paths:
                    nrof_image_total += 1
                    filename = os.path.splittext(os.path.split(image_path)[1])[0]
                    output_filename = os.path.join(output_class_dir, filename + '.png')
                    print('Image: {}'.format(image_path))
                    if not os.path.exits(output_filename):
                        try:   
                            img = misc.imread(image_path)
                        except(IOError, ValueError, IndexError) as e:
                            errorMessage = '{} : {}'.format(image_path, e)
                            print(errorMessage)
                        else:
                            if img.ndim == 2:
                                img = util.to_rgb(img)
                                print('to_rgb data dimention: ', img.ndim)
                            img = img[:, :, 0:3]
                            