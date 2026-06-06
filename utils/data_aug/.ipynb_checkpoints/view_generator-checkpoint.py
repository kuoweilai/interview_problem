
'''

https://github.com/sthalles/SimCLR/blob/master/data_aug/view_generator.py

'''
import numpy as np

np.random.seed(0)


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform
        

    def __call__(self, x):
        return self.base_transform(x)