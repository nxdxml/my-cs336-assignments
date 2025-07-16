git clone后需要增加的操作
1. 下载data
2. 文件夹路径很多写死了，在一些文件的非官方测试代码和train.py的配置等地方
3. 修改checkpoint dataset.npy等文件的保存路径

安装uv管理环境
参考 https://github.com/astral-sh/uv
可能需要添加环境变量
pyproject.toml：定义项目的主要依赖，包括项目名称、版本、描述、支持的 Python 版本等信息
``` sh
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```


initialize a project in the working directory:
``` sh
uv init
```

测试函数，函数模板在tests\adapters.py
```
uv run <python_file_path>
```

下载数据
``` sh
mkdir -p data
cd data

# linux
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

# windows
Invoke-WebRequest -Uri "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt" -OutFile "TinyStoriesV2-GPT4-train.txt"
Invoke-WebRequest -Uri "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt" -OutFile "TinyStoriesV2-GPT4-valid.txt"
```

运行单个测试文件
```
uv run pytest tests/test_tokenizer.py
uv run pytest tests/test_tokenizer.py::test_roundtrip_single_character
```

Python 不允许脚本文件使用相对导入，除非它被当作模块运行。使用指定PYTHONPATH+绝对导入
```
$env:PYTHONPATH = "."; uv run cs336_basics/BPETokenizer.py
```

训练
```
$env:PYTHONPATH = "."
uv run cs336_basics\train.py
```

参考
环境问题参考
https://blog.csdn.net/Humbunklung/article/details/146046406


