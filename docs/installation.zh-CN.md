# 安装与配置

[← README](../README.zh-CN.md) · [English](installation.md)

本文介绍本地模型服务、API 后端和部署选项。除特别说明外，所有命令都在 LightDubber 仓库根目录运行。

## 环境安装

需要 Linux、CUDA GPU、FFmpeg、`curl`、`unzip` 和 [`uv`](https://docs.astral.sh/uv/getting-started/installation/)。

```bash
# 主管线 + 本地 Qwen3-TTS
bash scripts/install.sh

# 同时安装本地 LightLLM 翻译服务
bash scripts/install.sh --translation lightllm

# 使用外部 TTS API，不安装本地 TTS 环境
bash scripts/install.sh --tts api
```

按部署方式选择一条命令。只将 TTS 放在远端时，可以组合 `--tts api --translation lightllm`。

| 环境 | 用途 | 何时安装 |
| :-- | :-- | :-- |
| `.venv` | 识别、分离、视觉绑定、管线和网页 | 始终安装 |
| `.venv-tts` | 本地 Qwen3-TTS | 默认安装，或指定 `--tts local` |
| `.venv-lightllm` | 本地翻译服务 | 指定 `--translation lightllm` |
| 现有 LightX2V 环境 | TBDub 口型编辑服务 | 单独准备 |

**TTS 必须独立环境吗？** 当前的本地 Qwen3-TTS 需要：0.1.1 版本依赖 Transformers 4.57.3，而主管线的 ASR、对齐和说话人分离模型使用更新的 Transformers 接口。使用主管线版本导入 Qwen3-TTS 时，`check_model_inputs` 装饰器因缺少 `func` 参数报错；反过来降到 4.57.3，则无法识别 ASR、对齐和说话人分离的模型类型。安装脚本自动处理环境隔离，管线通过配置好的解释器在 worker 所用 GPU 上启动 TTS，无需手动激活或切换环境。使用 API TTS 不需要本地 TTS 环境。

主管线安装脚本还会链接媒体解码所需的 FFmpeg 库。LightLLM 的 GPU 和共享内存要求见[官方安装文档](https://lightllm-en.readthedocs.io/en/latest/getting_started/installation.html)。

<a id="model-downloads"></a>
## 模型下载

```bash
export MODEL_DIR=/path/to/models/video_dubbing
PYTHON_BIN="$PWD/.venv/bin/python" bash scripts/download_models.sh all
```

需要鉴权的模型可通过 `.venv/bin/hf auth login` 登录。SAM-Audio Large-TV 默认从 ModelScope 下载；设置 `SAM_AUDIO_DOWNLOAD_SOURCE=hf` 可改用 Hugging Face。

| 组件组 | 模型 |
| :-- | :-- |
| `core` | TBDub Student、Wan VAE、HuBERT、人脸检测/几何/身份模型、LoCoNet |
| `recognition` | Qwen3-ASR、Qwen3-ForcedAligner、Nemotron 说话人分离 |
| `speech` | 识别模型和 Qwen3-TTS |
| `loconet` | UniTalk-LoCoNet 活跃说话人权重 |
| `sam-audio` | SAM-Audio Large-TV 及文本编码器 |
| `qwen35-translation` | Qwen3.5-9B |
| `all` | 上述全部模型 |

将 `all` 替换为组件名即可单独下载。翻译和 TTS 都使用 API 时，只需下载 `core`、`recognition` 和 `sam-audio`，不需要本地翻译和 TTS 权重。

生成本地配置：

```bash
.venv/bin/python scripts/configure.py \
  --models "$MODEL_DIR" --output config.local.json
```

模板是 [config.example.json](../config.example.json)。生成的文件包含本地绝对路径，不会被 Git 跟踪。复用已有 TTS 环境时，可指定 `--tts-python /path/to/python`；需要覆盖已有配置时才使用 `--force`。

<a id="model-services"></a>
## 模型服务

本地部署使用独立的翻译和口型编辑服务。先在不同终端启动这两个服务，再启动网页或 CLI。每个终端的 `MODEL_DIR` 应指向同一模型目录。

| GPU 分配示例 | 进程 |
| :-- | :-- |
| 0 | LightLLM 翻译 |
| 1 | LightX2V 口型编辑 |
| 2、3 | LightDubber 管线 worker；本地 TTS 与对应 worker 共用 GPU |

上述卡号用于说明进程分配，请根据模型、显存和所需并发量调整。

### 翻译 · LightLLM

使用 `--translation lightllm` 安装完成后：

```bash
GPU_ID=0 MODEL_PATH="$MODEL_DIR/Qwen3.5-9B" \
  bash scripts/serve_translation.sh
```

脚本使用 `.venv-lightllm/bin/python`；可通过 `LIGHTLLM_PYTHON` 指定已有安装。默认 `translation_backend` 是 LightLLM，模型服务名为 `qwen35-dubbing`，地址为 `http://127.0.0.1:18440/v1`。服务脚本读取 `config.local.json`，也可用 `DUB_CONFIG` 指定配置。

<details>
<summary>可选：本地 vLLM</summary>

```bash
uv venv --python 3.11 .venv-vllm
uv pip install --python .venv-vllm/bin/python vllm==0.23.0
```

在 `config.local.json` 中将 `translation_backend` 设为 `vllm`，再启动：

```bash
GPU_ID=0 MODEL_PATH="$MODEL_DIR/Qwen3.5-9B" \
  VLLM_BIN="$PWD/.venv-vllm/bin/vllm" bash scripts/serve_translation.sh
```

`TRANSLATION_BACKEND` 只覆盖服务启动脚本，适合临时测试。正式切换时应修改配置，让服务、管线和网页使用相同后端。

</details>

### 口型编辑 · LightX2V

按 [LightX2V](https://github.com/ModelTC/LightX2V) 仓库说明安装支持原生 TBDub 的版本；其 Python 环境还需要 `scikit-image` 进行颜色校正。然后启动：

```bash
GPU_ID=1 LIGHTX2V_PATH=/path/to/LightX2V \
  PYTHON_BIN=/path/to/lightx2v-env/bin/python \
  MODEL_PATH="$MODEL_DIR/tbdub" \
  CONFIG_PATH="$PWD/configs/tbdub_server.json" \
  bash scripts/serve_lipsync.sh
```

默认地址为 `http://127.0.0.1:18441`，对应配置中的 `tbdub_url`。**LightX2V 和 LightDubber 必须能通过相同的绝对路径访问媒体文件**，跨容器部署时也一样。网页调用这些内部服务，不需要把模型端口暴露到公网。

<a id="api-backends"></a>
## 翻译与 TTS API

翻译和 TTS 可以独立使用 OpenAI 兼容服务。**翻译可直接调用 DeepSeek 或 OpenAI GPT，无需安装 LightLLM/vLLM、下载 Qwen3.5，也不占本地翻译 GPU。** 其他管线阶段和 LightX2V 口型编辑服务仍在本地运行。

| 部署方式 | 安装命令 | 需要下载的模型组 |
| :-- | :-- | :-- |
| 翻译 API + 本地声音克隆 | `bash scripts/install.sh` | `core`、`speech`、`sam-audio` |
| 翻译 API + TTS API | `bash scripts/install.sh --tts api` | `core`、`recognition`、`sam-audio` |

按前文设置 `MODEL_DIR`，再用 `PYTHON_BIN="$PWD/.venv/bin/python" bash scripts/download_models.sh <group>` 分别下载所需组件。随后选择下面一种翻译服务，生成 `config.local.json`。如果已有配置，修改对应字段即可，不需要覆盖整个文件。

### DeepSeek 翻译

以下示例通过 DeepSeek 的 [OpenAI 兼容接口](https://api-docs.deepseek.com/en/) 调用 `deepseek-flash`。在运行 LightDubber 的进程环境中设置 `DEEPSEEK_API_KEY`。

```bash
.venv/bin/python scripts/configure.py \
  --models "$MODEL_DIR" --output config.local.json \
  --translation-backend openai_compatible \
  --translation-url https://api.deepseek.com \
  --translation-model deepseek-flash \
  --translation-api-key-env DEEPSEEK_API_KEY \
  --translation-options '{"thinking":{"type":"disabled"},"max_tokens":1024,"temperature":0}'
```

### OpenAI GPT 翻译

也可使用 [GPT-5.4 mini](https://developers.openai.com/api/docs/models/gpt-5.4-mini)。在服务环境中设置 `OPENAI_API_KEY` 后生成配置：

```bash
.venv/bin/python scripts/configure.py \
  --models "$MODEL_DIR" --output config.local.json \
  --translation-backend openai_compatible \
  --translation-url https://api.openai.com/v1 \
  --translation-model gpt-5.4-mini \
  --translation-api-key-env OPENAI_API_KEY \
  --translation-options '{"reasoning_effort":"none","max_completion_tokens":1024}'
```

两者均调用 `/chat/completions`。`translation_options` 用于传入不同服务需要的请求参数；客户端不会对所有 API 模型强制使用相同的温度或 token 上限字段。示例关闭推理模式以直接输出译文，可根据服务商调整模型名和参数。

### TTS API

上述示例保留本地 Qwen3-TTS，用参考音频克隆角色声音。如需改用 TTS API，安装时指定 `--tts api`，并将以下参数追加到选定的配置命令：

```bash
  --tts-backend openai_compatible \
  --tts-url https://tts.example.com/v1 \
  --tts-model your-tts-model --tts-voice alloy
```

已有配置可直接修改 `config.local.json` 中的这些字段：

```json
{
  "tts_backend": "openai_compatible",
  "tts_url": "https://tts.example.com/v1",
  "tts_model": "your-tts-model",
  "tts_voice": "alloy",
  "tts_api_key_env": "LIGHTDUBBER_TTS_API_KEY"
}
```

API TTS 使用 `/audio/speech` 接口和 WAV 输出，音色为服务商提供的预设音色，不会克隆源视频中的角色声音；需要声音克隆时保留本地 `qwen3tts`。音色名称以服务商支持的值为准。

密钥通过启动网页或 CLI 的环境变量传入，配置只保存变量名，不保存密钥本身；使用容器时也要将这些变量传入 worker 容器。翻译使用 API 时跳过本地翻译启动脚本，正常启动 LightX2V 和网页或 CLI 即可。

<a id="usage-options"></a>
## 使用选项

### 网页

```bash
DUB_GPUS=2,3 DUB_CONFIG="$PWD/config.local.json" \
  DUB_DATA_DIR="$PWD/outputs/web" bash scripts/serve.sh
```

worker 模型加载完成后，打开 **http://127.0.0.1:18320**。每张卡运行一个管线 worker，两个 worker 可并发处理两个任务；共享的口型服务对裁剪请求排队。单 worker 设置为 `DUB_GPUS=2`。

上传默认每个文件最大 **1 GB**，请求分块为 **8 MiB**，每 **2 MiB** 保存一次进度。中断后可从服务器已确认的位置续传。最多支持 16 个并发上传，与推理 worker 数无关。启动网页时可通过 `DUB_MAX_UPLOAD_MB`、`DUB_UPLOAD_CHUNK_MB` 和 `DUB_MAX_CONCURRENT_UPLOADS` 调整。

<details>
<summary>临时公网 HTTPS 入口</summary>

安装 `cloudflared` 后：

```bash
CLOUDFLARED_BIN=/path/to/cloudflared PORT=18320 bash scripts/public.sh
```

终端会打印临时 HTTPS 地址，持有链接的人可以使用服务；停止隧道即可取消公网访问。只公开网页端口，翻译和 LightX2V 服务保持本地访问。

</details>

### 命令行

使用空闲的管线 GPU，不要与正在运行的网页 worker 重复占用：

```bash
.venv/bin/python main.py \
  --config config.local.json --gpu 2 \
  --source input.mp4 --target-language en \
  --output outputs/example_en
```

| 参数 | 行为 |
| :-- | :-- |
| `--target-language en` | 必填目标语言：`zh/en/ja/ko/de/fr/ru/pt/es/it` |
| `--source-language zh` | 手动指定源语言，覆盖自动识别 |
| `--duration-mode adaptive_video` | 根据译音调整符合条件的视频与背景声片段；默认配置 |
| `--duration-mode fixed` | 保持视频时长，将译音适配到原时窗 |
| `--resume` | 续跑同一个请求；更改输入或配置后应使用新的输出目录 |

### 输出与日志

| 文件 | 内容 |
| :-- | :-- |
| `final.mp4` | 最终视频 |
| `final_fixed.mp4` | 自适应时长模式下保留的调整前视频 |
| `source.txt` / `translated.txt` | 识别的对白与译文 |
| `report.json` | 音频替换、口型处理结果，翻译与 TTS 分项耗时 |
| `run.json` / `logs/` | 管线进度与阶段日志 |

网页任务保存在 `$DUB_DATA_DIR/jobs/<id>/`，常驻模型加载日志保存在 `$DUB_DATA_DIR/resident/`。前台服务可用 `Ctrl+C` 停止。
