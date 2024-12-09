from typing import List, Dict, Type

from torch.utils.data import Dataset

from opencood.data_utils.data_models.dataset_models import CAVData
from opencood.data_utils.datasets.base_datasets.opv2v_dataset import OPV2VDataset


class DiffusionDataset(Dataset):
    def __init__(self, params, visualize, train, baseDataset: Type[OPV2VDataset]):
        super().__init__()
        self.baseDataset = baseDataset(params, visualize, train)
        # self.preprocess = transforms.Compose([transforms.Resize(64), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        self.baseDataset.reinitialize()

    def __len__(self):
        return self.baseDataset.__len__()

    def __getitem__(self, item):
        return self.baseDataset.GetNoisedData(item)

    def collate_batch_train(self, batch: List[Dict[str, CAVData]]):
        imgs = []
        for one_data in batch:
            for v in one_data.values():
                for camera in v.camera_data:
                    return camera
                    # return torch.stack(imgs, dim=0)
