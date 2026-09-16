import json


def dict_to_module_variables(config_dict, file_name="config.py"):
    """
    将字典转换为 Python 模块文件，字典的键作为模块级别的变量，值作为变量的值。

    :param config_dict: 包含配置的字典
    :param file_name: 生成的 Python 文件名称
    """
    with open(file_name, "w") as f:
        for key, value in config_dict.items():
            if isinstance(value, dict):
                # 如果值是嵌套的字典，使用 pprint.pformat 来格式化输出
                # formatted_value = pprint.pformat(value, indent=4)
                formatted_value = json.dumps(value, indent=2)
                formatted_value = formatted_value.replace("null", "None")
                formatted_value = formatted_value.replace("true", "True")
                formatted_value = formatted_value.replace("false", "False")
                f.write(f"{key} = {formatted_value}\n\n")
            else:
                # 对于非字典类型的值，直接写入
                f.write(f"{key} = {repr(value)}\n")


def indent(s_, num_spaces):
    s = s_.split("\n")
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(num_spaces * " ") + line for line in s]
    s = "\n".join(s)
    s = first + "\n" + s
    return s


def format_basic_types(k, v, use_mapping=False):
    if isinstance(v, str):
        v_str = f"'{v}'"
    else:
        v_str = str(v)

    if use_mapping:
        k_str = f"'{k}'" if isinstance(k, str) else str(k)
        attr_str = f"{k_str}: {v_str}"
    else:
        attr_str = f"{str(k)}={v_str}"
    attr_str = indent(attr_str, num_spaces=4)

    return attr_str


def _contain_invalid_identifier(dict_str):
    contain_invalid_identifier = False
    for key_name in dict_str:
        contain_invalid_identifier |= not str(key_name).isidentifier()
    return contain_invalid_identifier


def format_list(k, v, use_mapping=False):
    # check if all items in the list are dict
    if all(isinstance(_, dict) for _ in v):
        v_str = "[\n"
        v_str += "\n".join(f"dict({indent(format_dict(v_), num_spaces=4)})," for v_ in v).rstrip(
            ","
        )
        if use_mapping:
            k_str = f"'{k}'" if isinstance(k, str) else str(k)
            attr_str = f"{k_str}: {v_str}"
        else:
            attr_str = f"{str(k)}={v_str}"
        attr_str = indent(attr_str, num_spaces=4) + "]"
    else:
        attr_str = format_basic_types(k, v, use_mapping)
    return attr_str


def format_dict(input_dict, outest_level=False):
    r = ""
    s = []

    use_mapping = _contain_invalid_identifier(input_dict)
    if use_mapping:
        r += "{"
    for idx, (k, v) in enumerate(input_dict.items()):
        is_last = idx >= len(input_dict) - 1
        end = "" if outest_level or is_last else ","
        if isinstance(v, dict):
            v_str = "\n" + format_dict(v)
            if use_mapping:
                k_str = f"'{k}'" if isinstance(k, str) else str(k)
                attr_str = f"{k_str}: dict({v_str}"
            else:
                attr_str = f"{str(k)}=dict({v_str}"
            attr_str = indent(attr_str, num_spaces=4) + "\n)" + end
        elif isinstance(v, list):
            attr_str = format_list(k, v, use_mapping) + end
        else:
            attr_str = format_basic_types(k, v, use_mapping) + end

        s.append(attr_str)

    r += "\n".join(s)
    if use_mapping:
        r += "}"
    return r


def dict_to_file(config_dict, file_name="config.py"):
    with open(file_name, "a") as f:
        for key, val in config_dict.items():
            cur = {key: val}
            text = format_dict(cur) + "\n\n"
            f.write(text)


if __name__ == "__main__":

    exp = {
        "model": {
            "type": "hAlgorithm.modules.pipelines.depth_prompt_pointmap_pipeline.DepthPromptPointMapPipeline",
            "head": {
                "type": "hAlgorithm.modules.models.promptda.pointmap_dpt.PointMapDPTHead",
                "nclass": 1,
                "use_bn": False,
                "use_clstoken": False,
                "output_act": "",
                "with_uv": False,
            },
        }
    }

    dict_to_file(exp)
