from abc import ABC, abstractmethod

from lib.data_load import AttrDict, DataLoader


class ModelBase(ABC):

    @abstractmethod
    def train(
            self,
            args: AttrDict,
            data_loader: DataLoader,
            dev_data_loader: DataLoader = None,
            **kwargs,
    ):
        pass

    def restore(self, save_dir):
        pass

    def predict_batch(self, batch_frame_feat, batch_frame_len):
        pass
