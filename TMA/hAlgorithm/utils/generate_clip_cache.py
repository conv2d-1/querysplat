import json
import multiprocessing as mp
import os

import numpy as np


def calculate_scene_start_indices(datas):
    """预计算每个场景的起始索引和总长度"""
    start_indices = {}
    total_length = 0
    for scene in datas:
        scene_views_count = 0
        for frame in datas[scene]:
            views = frame["views"]
            scene_views_count += len(views)
        start_indices[scene] = total_length
        total_length += scene_views_count
    return start_indices, total_length


def process_scene(args):
    """处理单个场景的逻辑（进程工作函数）"""
    scene, frames, start_idx, mf_frame_num, mf_view_num = args

    # 生成当前场景的 frame_idxes
    frame_idxes = []
    current_scene_offset = start_idx
    for frame_idx, frame in enumerate(frames):
        views = frame["views"]
        views_count = len(views)
        frame_start = current_scene_offset + sum(len(f["views"]) for f in frames[:frame_idx])
        frame_indices = [frame_start + i for i in range(views_count)]
        frame_idxes.append(frame_indices)

    # 生成 clip 序列
    clip_seqs = generate_clip_sequence(scene, frames, frame_idxes, mf_frame_num, mf_view_num)

    return scene, clip_seqs


def generate_clip_cache_parallel(json_path, frame_num, view_num):
    with open(json_path, "r") as json_file:
        data_dict = json.load(json_file)["mf_files"]

    # 预计算场景起始索引
    scene_start_indices, _ = calculate_scene_start_indices(data_dict)

    # 创建任务列表
    tasks = []
    for scene in data_dict:
        frames = data_dict[scene]
        start_idx = scene_start_indices[scene]
        tasks.append((scene, frames, start_idx, frame_num, view_num))

    # 并行处理并显示进度条
    with mp.Pool() as pool:
        results = list(
            tqdm(
                pool.imap(process_scene, tasks),
                total=len(tasks),
                desc="Processing Scenes",
                ncols=100,
            )
        )

    # 合并结果并保存（后续代码不变）
    clip_infos = {}
    for scene, clip_seqs in results:
        if len(clip_seqs) > 0:
            clip_infos[scene] = clip_seqs

    output_dir = os.path.join(os.path.dirname(json_path), "clip_cache/")
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(json_path)
    filename = get_cach_name(filename, frame_num, view_num, tail="_mp")
    output_path = os.path.join(output_dir, filename)
    np.savez_compressed(output_path, **clip_infos)
    print(f"Saved to: {output_path}")


def generate_clip_cach(json_path, frame_num, view_num):
    with open(json_path, "r") as json_file:
        datas = json.load(json_file)["mf_files"]

    total_datas = []
    clip_infos = {}
    scene_infos = list(datas.keys())

    for scene in scene_infos:
        frames = datas[scene]

        frame_idxes = []
        for frame in frames:
            views = frame["views"]
            cur_id = len(total_datas)
            total_datas.extend(views)
            frame_idxes.append([cur_id + i for i, _ in enumerate(views)])

        clip_seqs = generate_clip_sequence(scene, frames, frame_idxes, frame_num, view_num)

        if len(clip_seqs) > 0:
            clip_infos[scene] = clip_seqs
        else:
            print(f"No Sequence for {scene}")

    output_dir = os.path.join(os.path.dirname(json_path), "clip_cache/")
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(json_path)
    filename = get_cach_name(filename, frame_num, view_num)
    output_path = os.path.join(output_dir, filename)
    np.savez_compressed(output_path, **clip_infos)
    print(output_path)


def get_cach_name(filename, frame_num, view_num, tail=""):
    name, ends = filename.split(".")
    return f"{name}_clip_cache_f{frame_num:04d}_v{view_num:04d}{tail}"


def generate_clip_sequence(scene, frames, frame_idxes, mf_frame_num, mf_view_num):
    seq_list = []
    seq_frame_id = [-f for f in range(mf_frame_num)]
    seq_frame_id.reverse()
    for i, frame in enumerate(frames):
        cur_seq_frame_id = [i + offset for offset in seq_frame_id]
        if min(cur_seq_frame_id) < 0:
            continue
        else:
            views = frame["views"]
            for j, view in enumerate(views):
                seq = []
                for view_i in range(mf_view_num):
                    cur_view_i = ((j - view_i) + len(views)) % len(views)
                    seq.extend(
                        [
                            [
                                scene,
                                frames[seq_i]["frame_id"],
                                views[cur_view_i]["view_id"],
                                frame_idxes[seq_i][cur_view_i],
                            ]
                            for seq_i in cur_seq_frame_id
                        ]
                    )
                seq_list.append(seq)
    return seq_list


def load_clip_sequence(json_path, frame_num, frame_step, view_num):
    # load cach
    filename = os.path.basename(json_path).split(".")[0]
    cach_dir = os.path.dirname(json_path)
    cach_list = os.listdir(cach_dir)
    cach_list = [f for f in cach_list if "clip_cache" in f and f.startswith(filename)]

    clip_frame_len = (frame_num - 1) * frame_step + 1
    clip_view_len = view_num

    valid_cach, cach_f, cach_v = choose_cach(cach_list, clip_frame_len, clip_view_len)
    if valid_cach is None:
        raise FileNotFoundError(
            f"Cach File Not Found for clip f:{clip_frame_len} v:{clip_view_len}"
        )
    assert cach_f and cach_v

    print(f"Found cach {valid_cach}")
    clip_seqs = np.load(os.path.join(cach_dir, valid_cach))

    # load data
    with open(json_path, "r") as json_file:
        datas = json.load(json_file)["mf_files"]

    total_datas = []
    clip_infos = []
    mf_infos = []
    scene_infos = list(datas.keys())

    mf_scene_sampling_strategy = "end:2"
    scene_sampling_number = 2

    if scene_sampling_number is not None:
        if mf_scene_sampling_strategy.startswith("first"):
            sample_scene_infos = scene_infos[:scene_sampling_number]
        elif mf_scene_sampling_strategy.startswith("end"):
            sample_scene_infos = scene_infos[-scene_sampling_number:]
        elif mf_scene_sampling_strategy.startswith("index"):
            sample_scene_infos = scene_infos[scene_sampling_number : scene_sampling_number + 1]

    start_idx = 0
    for scene in scene_infos:
        if scene == sample_scene_infos[0]:
            break
        else:
            frames = datas[scene]
            for frame in frames:
                views = frame["views"]
                start_idx += len(views)
    for scene in sample_scene_infos:
        frames = datas[scene]
        for frame in frames:
            frame_id = frame["frame_id"]
            views = frame["views"]
            total_datas.extend(views)
            mf_infos.extend([[scene, frame_id, view["view_id"]] for view in views])
        if len(frames) < cach_f:
            continue
        clips = clip_seqs[scene].tolist()
        for i, clip in enumerate(clips):
            sample_clip = []
            for v_i in range(view_num):
                for f_i in range(frame_num):
                    cach_id = v_i * cach_f + f_i * frame_step
                    cur_item = clip[cach_id]
                    cur_item[-3:] = [int(item) for item in cur_item[-3:]]
                    cur_item[-1] = cur_item[-1] - start_idx
                    sample_clip.append(cur_item)
            clips[i] = sample_clip
        clip_infos.extend(clips)
    return total_datas


def choose_cach(cach_list, clip_frame_len, clip_view_len):
    cach_list.sort()
    for cach_name in cach_list:
        cach_f, cach_v = cach_name.split(".")[0].split("_clip_cache_f")[1].split("_v")
        cach_f, cach_v = int(cach_f), int(cach_v)
        print(cach_name, cach_f, cach_v)
        if cach_f >= clip_frame_len and cach_v >= clip_view_len:
            return cach_name, cach_f, cach_v
    return None, None, None


def process_mission(info):
    mf_json, frame_num, view_num = info
    generate_clip_cache_parallel(mf_json, frame_num, view_num)


if __name__ == "__main__":

    import multiprocessing as mp

    from tqdm import tqdm

    json_list = [
        # "/mnt/netdata/Team/AI/datasets/TMD/habitat/train_mf.json",
        "/mnt/netdata/Team/AI/datasets/TMD/Habitat-Sim/train_mf.json",
        # "/mnt/netdata/Team/AI/datasets/TMD/DynamicStereo/train_mf.json",
        # "/mnt/netdata/Team/AI/datasets/TMD/Kubric-4D/test_mf.json",
        # "/mnt/netdata/Team/AI/datasets/TMD/Kubric-4D/tiny_mf.json",
        # "/mnt/netdata/Team/AI/datasets/TMD/Kubric-4D/train_mf.json",
        # "/mnt/netdata/Team/AI/datasets/TMD/Kubric-4D/val_mf.json",
    ]

    clip_configs = [(5, 1), (10, 1), (30, 1), (300, 1)]

    mission_list = []
    for mf_json in json_list:
        for frame_num, view_num in clip_configs:
            mission_list.append((mf_json, frame_num, view_num))
    for mission in mission_list:
        process_mission(mission)

    # num_processes = min(mp.cpu_count(), len(mission_list))
    # with mp.Pool(processes=num_processes) as pool:
    #     result_iter = pool.imap_unordered(process_mission, mission_list)
    #     for _ in tqdm(result_iter, total=len(mission_list), desc="Processing"):
    #             pass  # 忽略结果
