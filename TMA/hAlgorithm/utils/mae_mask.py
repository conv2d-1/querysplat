import numpy as np


class Cell:
    def __init__(self, num_masks, num_patches):
        self.num_masks = num_masks
        self.num_patches = num_patches
        self.size = num_masks + num_patches
        self.queue = np.hstack([np.ones(num_masks), np.zeros(num_patches)])
        self.queue_ptr = 0

    def set_ptr(self, pos=-1):
        self.queue_ptr = np.random.randint(self.size) if pos < 0 else pos

    def get_cell(self):
        cell_idx = (np.arange(self.size) + self.queue_ptr) % self.size
        return self.queue[cell_idx]

    def run_cell(self):
        self.queue_ptr += 1


def encoder_mask_map_generator(input_size, mask_ratio, mask_type="tube"):
    assert mask_type == "tube"

    frames, height, width = input_size
    num_patches_per_frame = height * width
    # total_patches = frames * num_patches_per_frame
    num_masks_per_frame = int(mask_ratio * num_patches_per_frame)
    # total_masks = self.frames * num_masks_per_frame
    mask_per_frame = np.hstack(
        [
            np.zeros(num_patches_per_frame - num_masks_per_frame),
            np.ones(num_masks_per_frame),
        ]
    )
    np.random.shuffle(mask_per_frame)
    mask = np.tile(mask_per_frame, (frames, 1))  # .flatten()

    return mask


def decoder_mask_map_generator(input_size, mask_ratio, mask_type="run_cell"):
    assert mask_type == "run_cell"

    frames, height, width = input_size

    num_masks_per_cell = int(4 * mask_ratio)
    assert 0 < num_masks_per_cell < 4
    num_patches_per_cell = 4 - num_masks_per_cell

    cell = Cell(num_masks_per_cell, num_patches_per_cell)
    cell_size = cell.size

    mask_list = []
    for ptr_pos in range(cell_size):
        cell.set_ptr(ptr_pos)
        mask = []
        for _ in range(frames):
            cell.run_cell()
            mask_unit = cell.get_cell().reshape(2, 2)
            mask_map = np.tile(mask_unit, [height // 2, width // 2])
            mask.append(mask_map.flatten())
        mask = np.stack(mask, axis=0)
        mask_list.append(mask)
    all_mask_maps = np.stack(mask_list, axis=0)

    return all_mask_maps
