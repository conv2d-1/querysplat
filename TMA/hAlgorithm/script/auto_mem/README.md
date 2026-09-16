- 每5分钟自动检查显存占用，如果有空卡会启动 60G 显存并占用 100% 的进程
```
python hAlgorithm/script/auto_mem/auto_mem.py
```

- 清理某张卡的自动占用
```
python hAlgorithm/script/auto_mem/clear_gpu.py [device]
```