# -*- coding: utf-8 -*-
import copy as deepcopy
import os
import sys
import tempfile
import time
from multiprocessing import Pool, cpu_count

import gradio as gr
import numpy as np
import open3d as o3d
import rerun as rr
import rerun.blueprint as rrb
import trimesh
from gradio_rerun import Rerun

os.environ["GRADIO_TEMP_DIR"] = f"/tmp/gradio_{time.time()}/"

rr.init("App")

# blueprint = rrb.Blueprint(
#     rrb.Horizontal(
#         rrb.Spatial3DView(),
#     ),
#     collapse_panels=False,
# )
# rr.send_blueprint(blueprint)


def get_ply_list(img_dir, filter_names):
    if os.path.isfile(img_dir):
        return [img_dir]

    imgs_list = []
    for root, dirs, files in os.walk(img_dir):
        for file in files:
            if file.lower().endswith(".ply"):
                path = os.path.join(root, file)
                if len(filter_names) > 0 and len(filter_names[0]) > 0:
                    if file in filter_names:
                        imgs_list.append(path)
                    else:
                        for name in filter_names:
                            if name in path:
                                imgs_list.append(path)
                                break
                else:
                    imgs_list.append(path)
    return sorted(imgs_list)


def streaming_mesh(state):
    stream = rr.binary_stream()

    current_glb_path, current_ply_path = state["paths"][state["current_index"]]
    pcd = o3d.io.read_point_cloud(current_ply_path)

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(),
        ),
        collapse_panels=False,
    )
    rr.send_blueprint(blueprint)

    rotation_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])

    for i in range(1):
        rr.set_time_sequence("frame", i)

        rr.log(
            "app/pointmap",
            rr.Points3D(
                pcd.points @ rotation_matrix.T,
                colors=pcd.colors,
            ),
        )

        yield stream.read()


def list2str(paths_list, current_index):
    paths_list = [f"{i}.{path}" for i, path in enumerate(paths_list)]
    paths_list[current_index] = "<b>" + "->" + paths_list[current_index] + "</b>"
    # cur_str = "\n".join(paths_list)
    cur_str = "<br>".join(paths_list)

    return cur_str


def update_gallery(directory, filter_names, current_index=0):
    print(directory)
    print(filter_names)
    filter_names = [name.strip() for name in filter_names.split(",") if name.strip()]
    paths_list = get_ply_list(directory, filter_names)

    if not paths_list:
        return None, {"paths": [], "current_index": 0}, "", ""

    # 更新状态以包含所有路径列表和当前索引
    state = {"paths": paths_list, "current_index": current_index}
    current_ply_path = state["paths"][state["current_index"]]

    stream = rr.binary_stream()

    current_ply_path = state["paths"][state["current_index"]]
    pcd = o3d.io.read_point_cloud(current_ply_path)

    rotation_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])

    rr.set_time_sequence("frame", 0)
    rr.log(
        "app/pointmap",
        rr.Points3D(
            pcd.points @ rotation_matrix.T,
            colors=pcd.colors,
        ),
    )

    yield stream.read(), state, current_ply_path, list2str(paths_list, current_index)


def next_frame(state):
    state["current_index"] = (state["current_index"] + 1) % len(state["paths"])
    current_ply_path = state["paths"][state["current_index"]]

    stream = rr.binary_stream()

    current_ply_path = state["paths"][state["current_index"]]
    pcd = o3d.io.read_point_cloud(current_ply_path)

    rotation_matrix = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])

    rr.set_time_sequence("frame", 0)
    rr.log(
        "app/pointmap",
        rr.Points3D(
            pcd.points,  # @ rotation_matrix.T,
            colors=pcd.colors,
        ),
    )

    yield stream.read(), state, current_ply_path, list2str(state["paths"], state["current_index"])


def prev_frame(state):
    state["current_index"] = (state["current_index"] - 1) % len(state["paths"])
    current_ply_path = state["paths"][state["current_index"]]

    stream = rr.binary_stream()

    current_ply_path = state["paths"][state["current_index"]]
    pcd = o3d.io.read_point_cloud(current_ply_path)

    rotation_matrix = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])

    rr.set_time_sequence("frame", 0)
    rr.log(
        "app/pointmap",
        rr.Points3D(
            pcd.points,  # @ rotation_matrix.T,
            colors=pcd.colors,
        ),
    )

    yield stream.read(), state, current_ply_path, list2str(state["paths"], state["current_index"])


with gr.Blocks() as demo:
    gr.Markdown("# 显示某路径下的所有图片的缩略图")
    assert len(sys.argv) == 2

    with gr.Row():
        text_input = gr.Textbox(interactive=True, value=sys.argv[1], label="输入ply目录")
        filter_names = gr.Textbox(label="过滤:name1,name2,", value="", interactive=True)

    with gr.Row():
        with gr.Column():
            path_display = gr.Textbox(label="当前显示的PLY路径", interactive=False)

        with gr.Column():
            btn_prev = gr.Button("上一帧")
            btn_next = gr.Button("下一帧")

    viewer = Rerun(
        streaming=False,
        panel_states={
            "time": "hidden",
            "blueprint": "hidden",
            "selection": "hidden",
        },
    )
    # total_paths = gr.Textbox(label="所有PLY文件", interactive=False)
    total_paths = gr.HTML(label="所有PLY文件")

    state = gr.State({"paths": [], "current_index": 0})

    text_input.change(
        fn=update_gallery,
        inputs=[text_input, filter_names],
        outputs=[viewer, state, path_display, total_paths],
    )
    filter_names.change(
        fn=update_gallery,
        inputs=[text_input, filter_names],
        outputs=[viewer, state, path_display, total_paths],
    )

    btn_prev.click(fn=prev_frame, inputs=state, outputs=[viewer, state, path_display, total_paths])
    btn_next.click(fn=next_frame, inputs=state, outputs=[viewer, state, path_display, total_paths])

    # 在应用启动时设置初始值
    def init_gallery():
        result = update_gallery(sys.argv[1], "")
        for output in result:
            yield output[0], output[1], output[2], output[3]

    demo.load(init_gallery, outputs=[viewer, state, path_display, total_paths])


if __name__ == "__main__":
    demo.launch(share=False)
