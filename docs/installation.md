# Installation and configuration

[← README](../README.md) · [简体中文](installation.zh-CN.md)

This guide covers local model services, API backends, and deployment options. Run commands from the LightDubber repository root unless stated otherwise.

## Environment setup

Requires Linux, CUDA GPUs, FFmpeg, `curl`, `unzip`, and [`uv`](https://docs.astral.sh/uv/getting-started/installation/).

```bash
# Pipeline + local Qwen3-TTS
bash scripts/install.sh

# Also install the local LightLLM translation server
bash scripts/install.sh --translation lightllm

# Use an external TTS API: omit the local TTS environment
bash scripts/install.sh --tts api
```

Choose one command for your deployment. `--tts api` and `--translation lightllm` can be combined when only TTS is remote.

| Environment | Purpose | Installed when |
| :-- | :-- | :-- |
| `.venv` | Recognition, separation, visual binding, pipeline, web UI | Always |
| `.venv-tts` | Local Qwen3-TTS | Default, or `--tts local` |
| `.venv-lightllm` | Local translation server | `--translation lightllm` |
| Your LightX2V environment | TBDub lip-sync server | Set up separately |

**Does TTS need its own environment?** The current local Qwen3-TTS does: version 0.1.1 depends on Transformers 4.57.3, while the pipeline's ASR, aligner, and diarization models use newer Transformers interfaces. With the pipeline version, Qwen3-TTS import fails because the `check_model_inputs` decorator now requires a `func` argument; downgrading to 4.57.3 leaves the ASR, aligner, and diarization model types unrecognized. The installer manages this separation, and the pipeline launches TTS through the configured interpreter on the worker's GPU. You do not need to activate or switch environments manually. API TTS needs no local TTS environment.

The main installer also links the FFmpeg libraries needed by media decoding. See the [LightLLM installation guide](https://lightllm-en.readthedocs.io/en/latest/getting_started/installation.html) for its GPU and shared-memory requirements.

<a id="model-downloads"></a>
## Model downloads

```bash
export MODEL_DIR=/path/to/models/video_dubbing
PYTHON_BIN="$PWD/.venv/bin/python" bash scripts/download_models.sh all
```

Use `.venv/bin/hf auth login` for models that require authentication. SAM-Audio Large-TV downloads from ModelScope by default; set `SAM_AUDIO_DOWNLOAD_SOURCE=hf` to use Hugging Face.

| Component group | Models |
| :-- | :-- |
| `core` | TBDub Student, Wan VAE, HuBERT, face detection/geometry/identity, LoCoNet |
| `recognition` | Qwen3-ASR, Qwen3-ForcedAligner, Nemotron diarization |
| `speech` | Recognition models and Qwen3-TTS |
| `loconet` | UniTalk-LoCoNet active-speaker weights |
| `sam-audio` | SAM-Audio Large-TV and its text encoder |
| `qwen35-translation` | Qwen3.5-9B |
| `all` | All of the above |

Replace `all` with a component name to download only that group. If both translation and TTS use APIs, download `core`, `recognition`, and `sam-audio`; local translation and TTS weights are unnecessary.

Generate the local configuration:

```bash
.venv/bin/python scripts/configure.py \
  --models "$MODEL_DIR" --output config.local.json
```

The template is [config.example.json](../config.example.json). The generated file contains absolute local paths and is ignored by Git. Use `--tts-python /path/to/python` to reuse an existing TTS environment. Add `--force` only to replace an existing configuration.

<a id="model-services"></a>
## Model services

The local setup uses separate translation and lip-sync services. Start them in separate terminals, then start the web UI or CLI. Set `MODEL_DIR` to the same model directory in each terminal.

| Example GPU assignment | Process |
| :-- | :-- |
| 0 | LightLLM translation |
| 1 | LightX2V lip-sync |
| 2, 3 | LightDubber pipeline workers; local TTS shares each worker's GPU |

These IDs illustrate process placement. Adjust them for your models, available GPU memory, and desired concurrency.

### Translation · LightLLM

After installing with `--translation lightllm`:

```bash
GPU_ID=0 MODEL_PATH="$MODEL_DIR/Qwen3.5-9B" \
  bash scripts/serve_translation.sh
```

The script uses `.venv-lightllm/bin/python`; set `LIGHTLLM_PYTHON` to reuse another installation. LightLLM is the default `translation_backend`, serving the alias `qwen35-dubbing` at `http://127.0.0.1:18440/v1`. The service script reads `config.local.json`, or the file set by `DUB_CONFIG`.

<details>
<summary>Alternative: local vLLM</summary>

```bash
uv venv --python 3.11 .venv-vllm
uv pip install --python .venv-vllm/bin/python vllm==0.23.0
```

Set `translation_backend` to `vllm` in `config.local.json`, then start:

```bash
GPU_ID=0 MODEL_PATH="$MODEL_DIR/Qwen3.5-9B" \
  VLLM_BIN="$PWD/.venv-vllm/bin/vllm" bash scripts/serve_translation.sh
```

`TRANSLATION_BACKEND` overrides only the service launcher for testing. Change the configuration to keep the service, pipeline, and web UI aligned.

</details>

### Lip-sync · LightX2V

Install [LightX2V](https://github.com/ModelTC/LightX2V) with native TBDub support using that repository's instructions. Its Python environment also needs `scikit-image` for color correction. Then start:

```bash
GPU_ID=1 LIGHTX2V_PATH=/path/to/LightX2V \
  PYTHON_BIN=/path/to/lightx2v-env/bin/python \
  MODEL_PATH="$MODEL_DIR/tbdub" \
  CONFIG_PATH="$PWD/configs/tbdub_server.json" \
  bash scripts/serve_lipsync.sh
```

The default address is `http://127.0.0.1:18441`, matching `tbdub_url`. **LightX2V and LightDubber must see the same absolute media paths**, including across containers. The web UI uses these internal services; they do not need public endpoints.

<a id="api-backends"></a>
## Translation and TTS APIs

Translation and TTS can independently use OpenAI-compatible providers. **DeepSeek and OpenAI GPT can handle translation without installing LightLLM/vLLM, downloading Qwen3.5, or reserving a local translation GPU.** The other pipeline stages and the LightX2V lip-sync service still run locally.

| Deployment | Install command | Model groups to download |
| :-- | :-- | :-- |
| Translation API + local voice cloning | `bash scripts/install.sh` | `core`, `speech`, `sam-audio` |
| Translation API + TTS API | `bash scripts/install.sh --tts api` | `core`, `recognition`, `sam-audio` |

Download each group with `PYTHON_BIN="$PWD/.venv/bin/python" bash scripts/download_models.sh <group>`, with `MODEL_DIR` set as above. Then choose one of the following translation providers when creating `config.local.json`. If you already have a configuration, update its corresponding fields instead of replacing it.

### DeepSeek translation

This example uses `deepseek-flash` through DeepSeek's [OpenAI-compatible endpoint](https://api-docs.deepseek.com/en/). Set `DEEPSEEK_API_KEY` in the environment of the process that will run LightDubber.

```bash
.venv/bin/python scripts/configure.py \
  --models "$MODEL_DIR" --output config.local.json \
  --translation-backend openai_compatible \
  --translation-url https://api.deepseek.com \
  --translation-model deepseek-flash \
  --translation-api-key-env DEEPSEEK_API_KEY \
  --translation-options '{"thinking":{"type":"disabled"},"max_tokens":1024,"temperature":0}'
```

### OpenAI GPT translation

Alternatively, use [GPT-5.4 mini](https://developers.openai.com/api/docs/models/gpt-5.4-mini). Set `OPENAI_API_KEY` in the service environment, then configure:

```bash
.venv/bin/python scripts/configure.py \
  --models "$MODEL_DIR" --output config.local.json \
  --translation-backend openai_compatible \
  --translation-url https://api.openai.com/v1 \
  --translation-model gpt-5.4-mini \
  --translation-api-key-env OPENAI_API_KEY \
  --translation-options '{"reasoning_effort":"none","max_completion_tokens":1024}'
```

Both use `/chat/completions`. `translation_options` supplies provider-specific request fields; the client does not force a shared temperature or token-limit parameter on every API model. The examples disable reasoning for direct translation. Change the model ID and options to match your provider.

### TTS API

The examples above retain local Qwen3-TTS for reference-audio voice cloning. To use a TTS API instead, install with `--tts api` and append these arguments to your chosen configuration command:

```bash
  --tts-backend openai_compatible \
  --tts-url https://tts.example.com/v1 \
  --tts-model your-tts-model --tts-voice alloy
```

For an existing configuration, update these fields in `config.local.json`:

```json
{
  "tts_backend": "openai_compatible",
  "tts_url": "https://tts.example.com/v1",
  "tts_model": "your-tts-model",
  "tts_voice": "alloy",
  "tts_api_key_env": "LIGHTDUBBER_TTS_API_KEY"
}
```

API TTS uses `/audio/speech` with WAV output and a provider-defined preset voice. It does not clone source speakers; retain local `qwen3tts` for voice cloning. Use a voice supported by your provider.

Supply keys through the environment of the web or CLI process; configuration stores environment variable names, not key values. For containers, pass those variables into the worker container. Skip the local translation launcher when using a translation API, then start LightX2V and the web UI or CLI as usual.

<a id="usage-options"></a>
## Usage options

### Web UI

```bash
DUB_GPUS=2,3 DUB_CONFIG="$PWD/config.local.json" \
  DUB_DATA_DIR="$PWD/outputs/web" bash scripts/serve.sh
```

Open **http://127.0.0.1:18320** after worker model loading completes. Each GPU runs one pipeline worker. Two workers can process two jobs concurrently, while the shared lip-sync service queues crop requests. Use `DUB_GPUS=2` for one worker.

Uploads default to a **1 GB** limit per file, **8 MiB** requests, and saved progress every **2 MiB**. Interrupted uploads can resume from the last confirmed byte. Up to 16 uploads run concurrently, independently of inference worker count. Set `DUB_MAX_UPLOAD_MB`, `DUB_UPLOAD_CHUNK_MB`, and `DUB_MAX_CONCURRENT_UPLOADS` when starting the web service to change these values.

<details>
<summary>Temporary public HTTPS access</summary>

With `cloudflared` installed:

```bash
CLOUDFLARED_BIN=/path/to/cloudflared PORT=18320 bash scripts/public.sh
```

The terminal prints a temporary HTTPS URL. Anyone with the link can use the service; stop the tunnel to remove public access. Only expose the web port, keeping translation and LightX2V local.

</details>

### CLI

Use an idle pipeline GPU, not one occupied by a web worker:

```bash
.venv/bin/python main.py \
  --config config.local.json --gpu 2 \
  --source input.mp4 --target-language en \
  --output outputs/example_en
```

| Option | Behavior |
| :-- | :-- |
| `--target-language en` | Required target language: `zh/en/ja/ko/de/fr/ru/pt/es/it` |
| `--source-language zh` | Override automatic source-language detection |
| `--duration-mode fixed` | Keep video duration and fit translated speech into its time window |
| `--resume` | Resume the same request; use a new output directory after changing inputs or configuration |

### Results and logs

| File | Contents |
| :-- | :-- |
| `final.mp4` | Delivered video |
| `source.txt` / `translated.txt` | Recognized and translated dialogue |
| `report.json` | Audio replacement, lip-sync outcomes, separate translation/TTS timings |
| `run.json` / `logs/` | Pipeline progress and stage logs |

Web jobs are under `$DUB_DATA_DIR/jobs/<id>/`; resident model loading logs are under `$DUB_DATA_DIR/resident/`. Stop foreground services with `Ctrl+C`.
