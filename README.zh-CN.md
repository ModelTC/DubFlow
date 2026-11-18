<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/lightdubber-logo-v3-dark.png" />
    <img src="docs/assets/lightdubber-logo-v3.png" width="640" alt="LightDubber" />
  </picture>
</p>

<h3 align="center">让视频跨越语言，让每个角色保留自己的声音。</h3>

<p align="center">
  在同一条模块化管线中完成对白翻译、语音生成与口型编辑。<br />
  支持本地模型，也可以接入自己的翻译和 TTS API。
</p>

<p align="center">
  <a href="README.md">English</a> · <strong>简体中文</strong>
</p>

<p align="center">
  <a href="#quick-start"><img src="https://img.shields.io/badge/Python-3.11-234c41?style=flat-square" alt="Python 3.11" /></a>
  <a href="#features"><img src="https://img.shields.io/badge/Languages-10-477866?style=flat-square" alt="支持 10 种目标语言" /></a>
  <a href="https://github.com/ModelTC/LightX2V"><img src="https://img.shields.io/badge/Lip--sync-LightX2V-234c41?style=flat-square" alt="LightX2V 口型编辑" /></a>
</p>

<p align="center">
  <a href="#features">功能特性</a> ·
  <a href="#demo">效果展示</a> ·
  <a href="#pipeline">处理流程</a> ·
  <a href="#quick-start">快速开始</a> ·
  <a href="docs/installation.zh-CN.md">安装与 API 配置</a>
</p>

---

**LightDubber** 是面向单人口播与多人对话的视频翻译配音框架，串联语音识别、视听说话人绑定、翻译、语音合成、口型编辑和视频合成。各阶段拥有独立的模型适配层，方便替换模型或接入 API，无需重写整条管线。

<a id="features"></a>
## 功能特性

| | 功能说明 |
| :-- | :-- |
| **区分角色的配音** | 将对白关联到画面中的说话人，本地语音克隆使用对应角色的参考音频。 |
| **多语言输出** | 支持 10 种目标语言，默认自动识别源语言。 |
| **可替换的模型后端** | 本地翻译使用 LightLLM，口型编辑使用 LightX2V；翻译与 TTS 也支持 OpenAI 兼容 API。 |
| **自适应时长** | 将译音适配到原视频时窗，或根据译音时长调整符合条件的视频片段。 |
| **网页与命令行** | 上传视频、排队处理、查看结果，也可通过 CLI 调用同一条管线。 |

**支持语言：** 中文 · 英语 · 日语 · 韩语 · 德语 · 法语 · 俄语 · 葡萄牙语 · 西班牙语 · 意大利语

<a id="demo"></a>
## 效果展示

**中文 → 英语**

<table>
<thead>
<tr>
<th align="center" width="50%">原视频 · 中文</th>
<th align="center" width="50%">LightDubber · 英语</th>
</tr>
</thead>
<tbody>
<tr>
<td width="50%" align="center" valign="top">

https://github.com/user-attachments/assets/83d17d3e-f1da-455e-b217-6ded433ff221

</td>
<td width="50%" align="center" valign="top">

https://github.com/user-attachments/assets/ffc9e8b9-be30-41cf-aec9-19c4e5c4f54e

</td>
</tr>
</tbody>
</table>

<a id="pipeline"></a>
## 处理流程

<p align="center">
  <img src="docs/assets/lightdubber-pipeline-v2.png" width="1100" alt="LightDubber 流程：识别语音与说话人、分离对白与背景、通过本地模型或 API 翻译与合成语音、使用 LightX2V 编辑口型、混音合成输出。" />
</p>

**识别与绑定 → 音频分离 → 翻译 → 语音合成 → 口型编辑 → 视频合成。**

ASR 与对齐模型提取对白及时间信息，视觉说话人绑定将台词关联到角色。音频分离保留背景声，翻译与 TTS 生成目标语言对白。LightX2V 编辑选定的人脸区域，最后混音并合成视频。未绑定到合适可见人脸的对白只替换音频，不进行口型编辑。

<a id="quick-start"></a>
## 快速开始

需要 **Linux、CUDA GPU、FFmpeg、`curl`、`unzip` 和 `uv`**。以下示例使用本地翻译与 TTS。**翻译也可调用 DeepSeek、OpenAI GPT 等 OpenAI 兼容 API**，无需本地 LightLLM/vLLM 环境、Qwen3.5 权重和翻译 GPU，配置见 [API 安装与示例](docs/installation.zh-CN.md#api-backends)。

### 1. 安装环境

```bash
git clone https://github.com/ModelTC/LightDubber.git
cd LightDubber
bash scripts/install.sh --translation lightllm
```

安装脚本准备主管线和本地 TTS 环境；`--translation lightllm` 同时安装本地翻译服务。当前 Qwen3-TTS 与主管线需要的 Transformers 版本不同，因此本地 TTS 保留独立 Python 环境，由脚本自动管理。使用 API TTS 时可通过 `--tts api` 跳过这个环境。

口型编辑需要安装支持原生 TBDub 的 [LightX2V](https://github.com/ModelTC/LightX2V)。提交任务前，请完成[模型服务配置](docs/installation.zh-CN.md#model-services)。

### 2. 下载模型并生成配置

```bash
export MODEL_DIR=/path/to/models/video_dubbing
PYTHON_BIN="$PWD/.venv/bin/python" bash scripts/download_models.sh all
.venv/bin/python scripts/configure.py --models "$MODEL_DIR" --output config.local.json
```

[config.example.json](config.example.json) 列出了模型路径和后端选项，生成的本地配置不会被 Git 跟踪。单独下载组件或切换下载源，请参阅[模型下载](docs/installation.zh-CN.md#model-downloads)。

### 3. 启动模型服务

在独立终端中启动 [LightLLM 和 LightX2V](docs/installation.zh-CN.md#model-services)。文档示例使用 GPU 0 运行翻译、GPU 1 运行口型编辑、GPU 2–3 运行管线 worker；请根据机器调整卡号与模型路径。

| 服务 | 默认地址 |
| :-- | :-- |
| 翻译 · LightLLM | `http://127.0.0.1:18440/v1` |
| 口型编辑 · LightX2V | `http://127.0.0.1:18441` |

### 4. 翻译视频

模型服务就绪后启动网页：

```bash
DUB_GPUS=2,3 DUB_CONFIG="$PWD/config.local.json" \
  DUB_DATA_DIR="$PWD/outputs/web" bash scripts/serve.sh
```

打开 **http://127.0.0.1:18320**，等待 worker 加载模型后，上传视频并选择目标语言。只启动一个管线 worker 时，设置 `DUB_GPUS=2`。

也可以在空闲的管线 GPU 上使用命令行：

```bash
.venv/bin/python main.py \
  --config config.local.json --gpu 2 \
  --source input.mp4 --target-language en \
  --output outputs/example_en
```

时长模式、上传、公网访问和任务续跑见[使用选项](docs/installation.zh-CN.md#usage-options)。

## 输出与开发

每个任务生成 `final.mp4`、`source.txt`、`translated.txt`、`report.json` 和运行日志；报告中分别统计翻译和 TTS 耗时。网页任务保存在 `$DUB_DATA_DIR/jobs/<id>/`。

<details>
<summary>目录结构与阶段接口</summary>

```text
main.py                  命令行入口
runtime/                 管线、配置、客户端与常驻模型进程
stages/
  prepare/               视频与音频标准化
  recognition/           ASR、对齐、声学说话人标签
  speakers/              人脸检测、活跃说话人、身份绑定
  separation/            对白与背景分离
  speech/                翻译与语音合成
  lipsync/               裁剪规划与 LightX2V 请求
  postprocess/           混音、回贴、时长调整与导出
web/                     上传、任务队列、结果查看
scripts/                 安装、下载与服务启动脚本
```

CLI 和网页 worker 共用 `Pipeline`。模型阶段在 `run.py` 暴露 `run_stage(...)`，在 `model.py` 放模型或服务适配，在 `utils.py` 放工具函数；导出阶段按音频、视频职责组织模块。`runtime/paths.py` 统一管理任务文件路径。S3FD、LoCoNet 和 SAM-Audio 推理代码放在使用它们的阶段目录内。

TTS 对相同请求复用已有音频，并停止不再改善时长的缩短重试。人脸检测缓存固定形状的先验框，并行处理帧编码与检测。CPU 回归测试覆盖缓存、阶段调度和任务续跑：

```bash
.venv/bin/python -m unittest discover -s tests -v
```

</details>

## 致谢

项目基于 [LightX2V](https://github.com/ModelTC/LightX2V)、[LightLLM](https://github.com/ModelTC/LightLLM) 及 [SOURCES.json](SOURCES.json) 中列出的上游模型构建，原始许可证与声明保留在 [licenses/](licenses/) 中。本地翻译也可使用 [vLLM](https://github.com/vllm-project/vllm) 后端。
