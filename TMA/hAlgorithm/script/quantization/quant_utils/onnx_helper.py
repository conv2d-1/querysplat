import os, sys
sys.path.append(os.getcwd())

from itertools import islice
import cv2
import numpy as np
import pandas as pd
import onnx
import onnxruntime as ort
import onnxoptimizer
from onnxsim import simplify
import torch
from tqdm import tqdm

from hAlgorithm.modules.pipelines.outputs import DepthOutput
from hAlgorithm.utils import (
    instantiate_from_config,
)
from hAlgorithm.utils.util import eval_dict_to_text
from hAlgorithm.script.quantization.quant_utils.data_helper import (
    np_depth_to_point,
    filter_pointmap,
    normalize_depth,
    summary_eval,
)

def before_last_slash(s):
    index = s.rfind("/")
    return s[:index] if index != -1 else s

def remove_node_from_graph(graph, node):
    input_name = node.input[0]   # 通常QuantizeLinear有一个输入
    output_name = node.output[0]  # 输出名称

    # 更新所有引用该输出的其他节点的输入
    for other_node in graph.node:
        if other_node == node:
            continue  # 跳过自身
        for i, inp in enumerate(other_node.input):
            if inp == output_name:
                other_node.input[i] = input_name
                break  # 找到第一个匹配项即退出
    graph.node.remove(node)

def get_amax_map(onnx_model, to_search_modules=[], fix_quant_layers=[], non_quant_layers=[], except_layers=[]):
    '''
    to_search_modules: should be format [('module_name', 'node_matching_part'), ...]
    fix_quant_layers: same as to_search_modules in this function, should be format [...]
    non_quant_layers: modules to skip
    except_layers : modules in fix_quant_layers to skip
    priority: non_quant_layers > except_layers > fix_quant_layers
    '''
    to_search_amax_config = {}
    for initializer in onnx_model.graph.initializer:
        # skip the except layers
        skip = False
        for fix_quant_layer in fix_quant_layers:
            if fix_quant_layer in initializer.name:
                skip = True
        for except_layer in except_layers:
            if except_layer in initializer.name:
                skip = False
        for non_quant_layer in non_quant_layers:
            if non_quant_layer in initializer.name:
                skip = True
        if skip: continue

        # add to search layers
        for to_search_module in to_search_modules:
            if to_search_module[0] in initializer.name and to_search_module[1] in initializer.name:
                orig_scale = onnx.numpy_helper.to_array(initializer)
                to_search_amax_config[initializer.name] = orig_scale
    return to_search_amax_config

def modify_onnx_quant(onnx_model, fix_layers=[], amax_config={}, non_quant_layers=[]):
    to_keep_layers = fix_layers.copy()
    for amax_name in amax_config.keys():
        to_keep_layers.append(before_last_slash(before_last_slash(amax_name)))

    # deep copy onnx model
    model_bytes = onnx_model.SerializeToString()
    onnx_model_cur = onnx.load_model_from_string(model_bytes)

    # 收集需删除的节点
    quant_nodes = []
    dequant_nodes = []
    for node in onnx_model_cur.graph.node:
        skip = False
        for keep_layer in to_keep_layers:
            if keep_layer in node.name:
                skip = True
        for non_quant_layer in non_quant_layers:
            if non_quant_layer in node.name:
                skip = False
        if skip: continue
        if node.op_type == 'QuantizeLinear':
            quant_nodes.append(node)
        if node.op_type == 'DequantizeLinear':
            dequant_nodes.append(node)
    assert len(quant_nodes) == len(dequant_nodes)

    # delete quant nodes
    print(f"deleting {len(quant_nodes)} quant nodes ...")
    passes = ["eliminate_unused_initializer"]
    for idx in range(len(quant_nodes)-1, -1, -1):
        remove_node_from_graph(onnx_model_cur.graph, dequant_nodes[idx])
        remove_node_from_graph(onnx_model_cur.graph, quant_nodes[idx])
    onnx_model_cur = onnxoptimizer.optimize(onnx_model_cur, passes)

    # modify quant layers
    for idx, initializer in enumerate(onnx_model_cur.graph.initializer):
        if initializer.name in amax_config.keys():
            orig_scale = onnx.numpy_helper.to_array(initializer)
            orig_amax = orig_scale * 127
            new_amax = amax_config[initializer.name]
            new_scale = np.array(new_amax/127.0, dtype=orig_scale.dtype)
            print(f"change {initializer.name}, amax from {orig_amax} to {new_amax}, scale from {orig_scale} to {new_scale}")
            new_initializer = onnx.numpy_helper.from_array(new_scale, initializer.name)
            onnx_model_cur.graph.initializer[idx].MergeFrom(new_initializer)

    return onnx_model_cur

class AmaxSearcher:
    def __init__(self, onnx_wrapper, dataloader, data_parser, postprocessor, eval_metrics, evaluator, cfg):
        self.onnx_wrapper = onnx_wrapper
        self.dataloader = dataloader
        self.data_parser = data_parser
        self.postprocessor = postprocessor
        self.eval_metrics = eval_metrics
        self.evaluator = evaluator

        self.cfg = cfg
        self.search_config = self.cfg['quantization']['onnx_analyse']
        self.output_dir = self.search_config['output_dir']

        self.best_record = {"best_ratio": 1.0, "best_result": None}
        self.last_result = None
        self.all_results = []
        self.skip_layers = []
        self.best_amax_records = []

    def search_amax(self):
        # load given amax config
        best_amax_config_path = self.search_config['best_amax_config_path']
        if best_amax_config_path is not None:
            df_best_amax = pd.read_csv(best_amax_config_path, header=None)
            self.best_amax_config = {row[0]: row[1] for row in df_best_amax.itertuples(index=False)}
        else:
            self.best_amax_config = self.search_config['best_amax_config']

        self.search_config['orig_scales'] = get_amax_map(
            self.onnx_wrapper.onnx_model, 
            self.search_config['to_search_modules'], 
            self.search_config['fix_quant_layers'], 
            self.search_config['non_quant_layers']
        )

        self.onnx_wrapper.modify_onnx(
            self.search_config['fix_quant_layers'], 
            {},
            self.search_config['non_quant_layers'],
        )
        frame_eval_results = self.evaluator.eval(
            self.onnx_wrapper, 
            self.dataloader, 
            self.data_parser, 
            self.postprocessor, 
            self.eval_metrics
        )
        self.last_result = summary_eval(frame_eval_results, self.eval_metrics)

        config_str = "basic"
        result_item = {"config": config_str, **self.last_result}
        self.all_results.append(result_item)
        df = pd.DataFrame(self.all_results)
        df.to_csv(f"{self.output_dir}/amax_search.csv", index=False)

        for to_search_layer, orig_scale in self.search_config['orig_scales'].items():
            if to_search_layer in self.best_amax_config.keys():
                continue
            amax_config = self.best_amax_config.copy()
            self.best_record = {"best_ratio": 1.0, "best_result": None}
            self.search_amax_one_layer(to_search_layer, amax_config)
        df = pd.DataFrame(self.best_amax_records)
        df.to_csv(f"{self.output_dir}/best_amax_record.csv", index=False)

    def search_amax_one_layer(self, to_search_layer, amax_config):
        self.search_config['offset'] = -self.search_config['half_amax_offset_scale']
        self.search_amax_one_layer_one_direction(to_search_layer, amax_config, skip_orig=False)
        self.search_config['offset'] = -self.search_config['half_amax_offset_scale']
        self.search_amax_one_layer_one_direction(to_search_layer, amax_config, skip_orig=True)

        if self.best_record['best_result']['abs_relative_difference'] - self.last_result['abs_relative_difference'] < self.search_config['skip_layer_thres']:
            self.last_result = self.best_record['best_result']
            orig_scale = self.search_config['orig_scales'][to_search_layer]
            self.best_amax_config[to_search_layer] = orig_scale * 127 * self.best_record['best_ratio']
            df = pd.DataFrame([self.best_amax_config]).T
            df.to_csv(f"{self.output_dir}/best_amax.csv", index=True, header=False)

            config_str = "\n".join([f"{k}={v}" for k, v in self.best_amax_config.items()])
            result_item = {"config": config_str, **self.last_result}
            self.best_amax_records.append(result_item)
        else:
            self.skip_layers.append(to_search_layer)
            print(f"skip quantization of layer :\n {self.skip_layers}")
            df = pd.DataFrame([self.skip_layers])
            df.to_csv(f"{self.output_dir}/skip_layers.csv", index=True, header=False)

    def search_amax_one_layer_one_direction(self, to_search_layer, amax_config, skip_orig=False):
        offset = self.search_config['offset']
        oneside_num = self.search_config['half_search_points']
        orig_scales = self.search_config['orig_scales']
        orig_scale = orig_scales[to_search_layer]

        scale_ratios = np.linspace(1.0, 1.0+offset, num=oneside_num).tolist()
        if skip_orig:
            scale_ratios = scale_ratios[1:]

        for scale_ratio in scale_ratios:
            amax_config[to_search_layer] = orig_scale * 127 * scale_ratio

            self.onnx_wrapper.modify_onnx(
                self.search_config['fix_quant_layers'], 
                amax_config,
                self.search_config['non_quant_layers'],
            )

            frame_eval_results = self.evaluator.eval(
                self.onnx_wrapper, 
                self.dataloader, 
                self.data_parser, 
                self.postprocessor, 
                self.eval_metrics
            )
            eval_results = summary_eval(frame_eval_results, self.eval_metrics)

            config_str = "\n".join([f"{k}={v}" for k, v in amax_config.items()])
            result_item = {"config": config_str, **eval_results}
            self.all_results.append(result_item)
            df = pd.DataFrame(self.all_results)
            df.to_csv(f"{self.output_dir}/amax_search.csv", index=False)

            if self.best_record["best_result"] is None:
                self.best_record["best_result"] = eval_results
                self.best_record["best_ratio"] = scale_ratio
            else:
                if eval_results['abs_relative_difference'] < self.best_record["best_result"]['abs_relative_difference']:
                    self.best_record["best_result"] = eval_results
                    self.best_record["best_ratio"] = scale_ratio
                elif eval_results['abs_relative_difference'] == self.best_record["best_result"]['abs_relative_difference'] and \
                        eval_results['abs_difference'] < self.best_record["best_result"]['abs_difference']:
                    self.best_record["best_result"] = eval_results
                    self.best_record["best_ratio"] = scale_ratio
                else:
                    print(f"Early stop!")
                    break
        return self.best_record

class SingleLayerAnalyser:
    def __init__(self, onnx_wrapper, dataloader, data_parser, postprocessor, eval_metrics, evaluator, cfg):
        self.onnx_wrapper = onnx_wrapper
        self.dataloader = dataloader
        self.data_parser = data_parser
        self.postprocessor = postprocessor
        self.eval_metrics = eval_metrics
        self.evaluator = evaluator

        self.cfg = cfg
        self.analyse_config = self.cfg['quantization']['onnx_analyse']
        self.output_dir = self.analyse_config['output_dir']

        self.all_results = []

    def analyse_layers(self):
        self.analyse_config['orig_scales'] = get_amax_map(
            self.onnx_wrapper.onnx_model, 
            self.analyse_config['to_search_modules'], 
            self.analyse_config['fix_quant_layers'], 
            self.analyse_config['non_quant_layers']
        )

        self.onnx_wrapper.modify_onnx(
            self.analyse_config['fix_quant_layers'], 
            {},
            self.analyse_config['non_quant_layers'],
        )
        frame_eval_results = self.evaluator.eval(
            self.onnx_wrapper, 
            self.dataloader, 
            self.data_parser, 
            self.postprocessor, 
            self.eval_metrics
        )
        fp16_result = summary_eval(frame_eval_results, self.eval_metrics)
        config_str = "fp16"
        result_item = {"config": config_str, **fp16_result}
        self.all_results.append(result_item)
        df = pd.DataFrame(self.all_results)
        df.to_csv(f"{self.output_dir}/layer_analyse.csv", index=False)

        for to_analyse_layer, orig_scale in self.analyse_config['orig_scales'].items():
            self.onnx_wrapper.modify_onnx(
                self.analyse_config['fix_quant_layers'], 
                {to_analyse_layer: orig_scale * 127},
                self.analyse_config['non_quant_layers'],
            )
            frame_eval_results = self.evaluator.eval(
                self.onnx_wrapper, 
                self.dataloader, 
                self.data_parser, 
                self.postprocessor, 
                self.eval_metrics
            )
            eval_results = summary_eval(frame_eval_results, self.eval_metrics)

            config_str = to_analyse_layer
            result_item = {"config": config_str, **eval_results}
            self.all_results.append(result_item)
            df = pd.DataFrame(self.all_results)
            df.to_csv(f"{self.output_dir}/layer_analyse.csv", index=False)


class DequantAnalyse:
    def __init__(self, onnx_wrapper, dataloader, data_parser, postprocessor, eval_metrics, evaluator, cfg):
        self.onnx_wrapper = onnx_wrapper
        self.dataloader = dataloader
        self.data_parser = data_parser
        self.postprocessor = postprocessor
        self.eval_metrics = eval_metrics
        self.evaluator = evaluator

        self.cfg = cfg
        self.analyse_config = self.cfg['quantization']['onnx_analyse']
        self.output_dir = self.analyse_config['output_dir']

        self.all_results = []

    def analyse(self):
        model_bytes = self.onnx_wrapper.onnx_model.SerializeToString()
        onnx_model = onnx.load_model_from_string(model_bytes)
        graph = onnx_model.graph
        nodes = graph.node

        quant_nodes = []
        dequant_nodes = []
        for node in nodes:
            if node.op_type == 'QuantizeLinear':
                quant_nodes.append(node)
            if node.op_type == 'DequantizeLinear':
                dequant_nodes.append(node)
        assert len(quant_nodes) == len(dequant_nodes)

        passes = ["eliminate_unused_initializer"]

        for idx in tqdm(range(len(quant_nodes)-1, -1, -1)):
            remove_node_from_graph(graph, dequant_nodes[idx])
            remove_node_from_graph(graph, quant_nodes[idx])
            print(f"remove node {dequant_nodes[idx].name}")
            print(f"remove node {quant_nodes[idx].name}")
            model_bytes = onnx_model.SerializeToString()
            onnx_model_cur = onnx.load_model_from_string(model_bytes)
            onnx_model_cur = onnxoptimizer.optimize(onnx_model_cur, passes)

            self.onnx_wrapper.set_onnx_model(onnx_model_cur)

            frame_eval_results = self.evaluator.eval(
                self.onnx_wrapper, 
                self.dataloader, 
                self.data_parser, 
                self.postprocessor, 
                self.eval_metrics
            )
            eval_results = summary_eval(frame_eval_results, self.eval_metrics)

            dequant_layer_name = before_last_slash(quant_nodes[idx].name)
            eval_results['dequant_layer_name'] = dequant_layer_name
            self.all_results.append(eval_results)
        
            df = pd.DataFrame(self.all_results)
            df.to_csv(f"{self.output_dir}/dequant_analyse.csv", index=False) 

class OnnxModel:
    def __init__(self, onnx_path):
        self.onnx_path = onnx_path

        self.orig_onnx_model = onnx.load(self.onnx_path)
        onnx.checker.check_model(self.orig_onnx_model)

        self.orig_onnx_model, check = simplify(
            self.orig_onnx_model,
            perform_optimization=False
        )

        self.onnx_model = self.orig_onnx_model
        
        self.provider_options = [{"device_id": 0}, {}]  # 使用 GPU 0
        self.providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(
            self.onnx_model.SerializeToString(), 
            providers=self.providers, 
            provider_options=self.provider_options
        )

    def modify_onnx_by_cfg(self, onnx_analyse_cfg):
        # load given amax config
        best_amax_config_path = onnx_analyse_cfg['best_amax_config_path']
        if best_amax_config_path is not None:
            df_best_amax = pd.read_csv(best_amax_config_path, header=None)
            best_amax_config = {row[0]: row[1] for row in df_best_amax.itertuples(index=False)}
        else:
            best_amax_config = onnx_analyse_cfg['best_amax_config']

        self.onnx_model = modify_onnx_quant(
            self.orig_onnx_model, 
            onnx_analyse_cfg['fix_quant_layers'], 
            best_amax_config, 
            onnx_analyse_cfg['non_quant_layers']
        )

        print("---------------------------- exist quant layers -------------------------")
        for node in self.onnx_model.graph.node:
            if node.op_type == 'QuantizeLinear':
                print(node.name)
            if node.op_type == 'DequantizeLinear':
                print(node.name)

        self.session = ort.InferenceSession(
            self.onnx_model.SerializeToString(), 
            providers=self.providers, 
            provider_options=self.provider_options
        )

    def modify_onnx(self, fix_layers=[], amax_config={}, non_quant_layers=[]):
        self.onnx_model = modify_onnx_quant(
            self.orig_onnx_model, 
            fix_layers, 
            amax_config, 
            non_quant_layers
        )

        self.session = ort.InferenceSession(
            self.onnx_model.SerializeToString(), 
            providers=self.providers, 
            provider_options=self.provider_options
        )

    def set_onnx_model(self, onnx_model):
        self.onnx_model = onnx_model
        self.session = ort.InferenceSession(
            self.onnx_model.SerializeToString(), 
            providers=self.providers, 
            provider_options=self.provider_options
        )

    def __call__(self, input_data, meta_data=None):
        inputs = {
            "image": input_data['image'].cpu().numpy(),
            "prompt_depth": input_data['prompt_depth'].cpu().numpy()[:,-1:,:,:], #[:,-1:,:,:]
            "prompt_scale": input_data['prompt_scale'].cpu().numpy(),
        }

        outputs = self.session.run(None, inputs)
        pointmap_pred = torch.from_numpy(outputs[0]).cuda()
        confidence_pred = torch.from_numpy(outputs[1]).cuda()

        pointmap_pred = normalize_depth(pointmap_pred, input_data['prompt_scale'].cuda(), None)

        results = {
            "pointmap": pointmap_pred,
            "confidence": confidence_pred
        }
        return results

    def save(self, save_path):
        onnx.save(self.onnx_model, save_path)