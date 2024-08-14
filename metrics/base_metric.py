from abc import ABC, abstractmethod

class BaseMetric(ABC):

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def compute(self,references,predictions):
        pass
        