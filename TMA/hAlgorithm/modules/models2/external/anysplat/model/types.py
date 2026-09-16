from dataclasses import dataclass
from pathlib import Path

from jaxtyping import Float
from torch import Tensor

from .ply_export import export_ply


@dataclass
class Gaussians:
    means: Float[Tensor, "batch gaussian dim"]
    covariances: Float[Tensor, "batch gaussian dim dim"]
    harmonics: Float[Tensor, "batch gaussian 3 d_sh"]
    opacities: Float[Tensor, "batch gaussian"]
    scales: Float[Tensor, "batch gaussian 3"]
    rotations: Float[Tensor, "batch gaussian 4"]
    # levels: Float[Tensor, "batch gaussian"]
    
    def export_ply(self, plyfile):
        export_ply(
            self.means[0],
            self.scales[0],
            self.rotations[0],
            self.harmonics[0],
            self.opacities[0],
            Path(plyfile),
            save_sh_dc_only=True,
        )