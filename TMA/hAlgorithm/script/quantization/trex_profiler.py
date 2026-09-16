import matplotlib.pyplot as plt
import os
import pandas as pd
import trex
import trex.notebook
import trex.plotting
import trex.graphing
import trex.df_preprocessing

import argparse

# Configure a wider output (for the wide graphs)
trex.notebook.set_wide_display()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quant your cute model!")
    parser.add_argument(
        "--engine",
        type=str,
        default=None,
        help="Path to engine file.",
    )
    args = parser.parse_args()
    
    engine_path = args.engine
    plan = trex.EnginePlan(f'{engine_path}.graph.json', f'{engine_path}.profile.json', f'{engine_path}.profile.metadata.json')

    formatter = trex.graphing.layer_type_formatter if True else trex.graphing.precision_formatter
    graph = trex.graphing.to_dot(plan, formatter)
    svg_name = trex.graphing.render_dot(graph, engine_path, 'svg')