import os
import sys

sys.path.append(os.getcwd())

from hAlgorithm.datasets.base_dataset import BaseDataset


class UrbansynDataset(BaseDataset):
    # cityscapes 19
    semantic_labels = dict(
        road=(128, 64, 128),
        sidewalk=(244, 35, 232),
        building=(70, 70, 70),
        wall=(102, 102, 156),
        fence=(190, 153, 153),
        pole=(153, 153, 153),
        traffic_light=(250, 170, 30),
        traffic_sign=(220, 220, 0),
        vegetation=(107, 142, 35),
        terrain=(152, 251, 152),
        sky=(70, 130, 180),
        person=(220, 20, 60),
        rider=(255, 0, 0),
        car=(0, 0, 142),
        trunk=(0, 0, 70),
        bus=(0, 60, 100),
        train=(0, 80, 100),
        motorcycle=(0, 0, 230),
        bicycle=(119, 11, 32),
        unlabeled=(0, 0, 0),
    )
    
    def __init__(
        self,
        name: str = "urbansyn",
        min_depth: float = 1e-3,
        max_depth: float = 150.0,
        sem_name: str  = "sem",
        depth_invalid_sem_names: list = ["sky"],
        **kwargs,
    ):
        super().__init__(
            name=name,
            min_depth=min_depth,
            max_depth=max_depth,
            sem_name=sem_name,
            depth_invalid_sem_names=depth_invalid_sem_names,
            **kwargs,
        )