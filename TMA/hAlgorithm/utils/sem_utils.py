import numpy as np

SEM_LABEL=dict(
    other=0,
    sky=1,
)

def remap_sem_label(curr_sem, ori_config: dict):
    if curr_sem is None:
        return None
    if len(curr_sem.shape) == 2:
        curr_sem = np.expand_dims(curr_sem, axis=-1)
    H, W, C = curr_sem.shape
    if ori_config is None:
        ori_config = {}
    config = {
        ori_config[k] : v for k, v in SEM_LABEL.items() if k in ori_config
    }
    try:
        keys = np.array([list(k) for k in config.keys()])
    except TypeError:
        keys = np.array([[k] for k in config.keys()])
    values = np.array(list(config.values()))

    # 构造输出数组，初始化为 other
    sem = np.full((H, W), SEM_LABEL['other'], dtype=np.int64)

    if len(keys) > 0:
        assert len(keys[0]) == C, f"ori_config channel {len(keys[0])}, but got {C} in curr_sem"
        
        # 广播比较
        matches = (curr_sem[:, :, None, :] == keys).all(axis=-1)  # shape: (H, W, N)

        # 找到第一个匹配的索引
        match_indices = matches.argmax(axis=-1)  # shape: (H, W)

        # 判断是否有匹配
        has_match = matches.any(axis=-1)


        # 设置匹配位置的值
        sem[has_match] = values[match_indices[has_match]]

    return sem

if __name__ == '__main__':
    curr_sem = np.zeros((200,200))
    curr_sem[:100, :100, ...] = 2
    sem = remap_sem_label(curr_sem, {"sky":0})
    print(sem)