import os

import torch

from .iter_trainer import IterTrainer


class GSInfer(IterTrainer):
    def __init__(self, **kwargs):

        model = kwargs["model"]
        val_datasets = kwargs["val_datasets"]
        device = torch.device("cuda")

        model.set_cameras(val_datasets[0].gs_total_datas, device=device)

        super().__init__(**kwargs)
    
    def register_schedule(self):
        pass

    def validate(self, save_best=False, save_tb=False, vis=False, save_outputs=False):
        os.makedirs(self.vis_dir, exist_ok=True)
        with torch.inference_mode():
            for i, data_loader in enumerate(self.val_dataloaders):
                batch = [data for data in data_loader]
                output = self.model.infer(batch)

                meta_data = [data["meta_data"] for data in batch]
                self.model.visualize(output, meta_data=meta_data, out_dir=self.vis_dir)

