import os
import sys

assert len(sys.argv) == 2

cmd = f"ps -ef | grep python | grep 'gpu_task.py {sys.argv[1]}'"
os.system(cmd)

cmd = (
    f"ps -ef | grep python | grep 'gpu_task.py {sys.argv[1]}' | awk '{{print $2}}' | xargs kill -9"
)
os.system(cmd)
