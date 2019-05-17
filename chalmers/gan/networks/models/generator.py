from abc import ABC, abstractmethod

class Generator:
    def __init__(self):
        super(Generator, self).__init__()

    @abstractmethod
    def __call__(self, image, is_training=False):
        pass

    def name(self):
        return self.__class__.__name__