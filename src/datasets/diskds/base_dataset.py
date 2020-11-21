from torch.utils.data import Dataset
from typing import List


class BaseDataset(Dataset):

    def __init__(self):
        super.__init__(self)

    def get_idx_list(self) -> List[str]:
        return []
