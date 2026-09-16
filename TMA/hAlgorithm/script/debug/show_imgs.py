# -*- coding: utf-8 -*-
import os
import time

import gradio as gr

os.environ["GRADIO_TEMP_DIR"] = f"/tmp/gradio_{time.time()}/"


def get_img_list(img_dir, filter_names):
    """
    递归获取指定目录及其所有子文件夹下所有图片文件的路径列表。
    """
    valid_extensions = (".png", ".jpg", ".jpeg", ".webp", ".tif")
    imgs_list = []

    for root, dirs, files in os.walk(img_dir):
        for file in files:
            if file.lower().endswith(valid_extensions):
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


def handle_select(evt: gr.SelectData, state):
    """
    处理Gallery选择事件，使用状态中的图片路径列表来确定选中的图片路径。
    """
    selected_index = evt.index
    selected_img_path = state[selected_index]
    return selected_img_path


# 创建Gradio界面
with gr.Blocks() as demo:
    gr.Markdown("# 显示某路径下的所有图片的缩略图")

    with gr.Row():
        text_input = gr.Textbox(interactive=True, value="", label="输入图片目录")
        filter_names = gr.Textbox(label="过滤:name1,name2,", value="", interactive=True)

    gallery_output = gr.Gallery(
        label="所有图片", columns=6, height="auto", show_download_button=True
    )
    path_display = gr.Textbox(label="图片路径", interactive=False)

    # 使用 gr.State 来保存状态
    state = gr.State({"img_paths": []})

    def update_gallery_and_handle_click(directory, filter_names):
        print(directory)
        print(filter_names)
        filter_names = filter_names.strip().split(",")
        img_paths_list = get_img_list(directory, filter_names)
        return img_paths_list, img_paths_list

    text_input.change(
        fn=update_gallery_and_handle_click,
        inputs=[text_input, filter_names],
        outputs=[gallery_output, state],
    )
    filter_names.change(
        fn=update_gallery_and_handle_click,
        inputs=[text_input, filter_names],
        outputs=[gallery_output, state],
    )
    gallery_output.select(fn=handle_select, inputs=[state], outputs=path_display)

if __name__ == "__main__":
    demo.launch(share=False)
