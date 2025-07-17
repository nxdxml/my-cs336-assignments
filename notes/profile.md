这里记录下对模型进行性能分析的一些记录，待补充

## nsight system profile
主要使用nvtx括住想要进行性能分析的部分，将生成的文件导入可视化窗口
1. 通过上面的图观察整体运行时间
2. 详细的核函数运行时间选择Stats System View ->  xxx kernel summary观察