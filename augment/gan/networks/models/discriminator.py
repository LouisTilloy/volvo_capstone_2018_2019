from abc import ABC, abstractmethod

class Discriminator:
    def __init__(self):
        super(Discriminator, self).__init__()

    @abstractmethod
    def __call__(self, image, reuse=False, is_training=False):
        pass
        
    def name(self):
        return self.__class__.__name__