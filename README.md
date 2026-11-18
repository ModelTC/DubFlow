<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/lightdubber-logo-v3-dark.png" />
    <img src="docs/assets/lightdubber-logo-v3.png" width="640" alt="LightDubber" />
  </picture>
</p>

<h3 align="center">End-to-end dubbing for multi-speaker, multi-shot video.</h3>

<p align="center">
  Translate conversations, clone speaker voices, and synchronize lips across shot changes.<br />
  Adapt video timing to translated speech and remix it with the original background sound.
</p>

<p align="center">
  <strong>English</strong> · <a href="README.zh-CN.md">简体中文</a>
</p>

<p align="center">
  <a href="#quick-start"><img src="https://img.shields.io/badge/Python-3.11-234c41?style=flat-square" alt="Python 3.11" /></a>
  <a href="#features"><img src="https://img.shields.io/badge/Languages-10-477866?style=flat-square" alt="10 target languages" /></a>
  <a href="https://github.com/ModelTC/LightX2V"><img src="https://img.shields.io/badge/Lip--sync-LightX2V-234c41?style=flat-square" alt="Lip-sync powered by LightX2V" /></a>
</p>

<p align="center">
  <a href="#features">Features</a> ·
  <a href="#demo">Demo</a> ·
  <a href="#pipeline">Pipeline</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="docs/installation.md">Installation & APIs</a>
</p>

---

**LightDubber** is an open-source video dubbing framework built for **multiple speakers, recurring characters, and changing shots**. It connects who said what to who appears on screen, then translates the conversation, generates dubbed speech, and synchronizes the corresponding faces. Choose to keep the original runtime or adapt eligible segments to the pace of the translated dialogue.

From interviews and dialogue scenes to single-speaker presentations, LightDubber brings translation, voices, lip motion, and timing into one automated workflow. Use local models or connect your own translation and TTS APIs; each stage can be replaced through a shared interface.

<a id="features"></a>
## Features

| Capability | What it enables |
| :-- | :-- |
| **Multiple speakers** | Associate dialogue with on-screen characters using voice, face identity, and visual speaking cues. |
| **Across shot changes** | Match recurring faces across shots and plan lip-sync edits within each shot's boundaries. |
| **Contextual translation** | Translate into 10 target languages using surrounding dialogue and speaker history as context, with automatic source-language detection. |
| **Voice cloning & lip sync** | Use each speaker's reference audio for local voice cloning, then drive the matched face with the translated speech through LightX2V. |
| **Adaptive duration** | Keep the original runtime, or extend and shorten eligible segments to fit the translated speech. |
| **Background ambience** | Separate dialogue from background audio, then remix the new voices with the source music and ambient sound. |
| **Local models or APIs** | Choose local inference or translation/TTS APIs independently, and replace individual models without rewriting the workflow. |
| **Web UI & CLI** | Upload videos, queue jobs, preview results, or automate the same workflow from the command line. |

**Languages:** Chinese · English · Japanese · Korean · German · French · Russian · Portuguese · Spanish · Italian

<a id="demo"></a>
## Demo

**Chinese → English**

<table>
<thead>
<tr>
<th align="center" width="50%">Original · Chinese</th>
<th align="center" width="50%">LightDubber · English</th>
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
## Pipeline

<p align="center">
  <img src="docs/assets/lightdubber-pipeline-v2.png" width="1100" alt="LightDubber pipeline: recognize speech and speakers, separate dialogue and background, translate using a local LLM or API, synthesize voices using local TTS or an API, edit lip motion with LightX2V, then composite and export." />
</p>

**Dialogue, voices, faces, and timing—all coordinated in one workflow.**

1. **Map the conversation across shots.** Detect shot boundaries, recover dialogue and timing, and associate spoken lines with recurring on-screen characters.
2. **Translate and voice each turn.** Use conversational context to translate each line, and generate the new speech with local voice cloning or your chosen TTS API.
3. **Synchronize the right face.** Drive the matched speaker's mouth with the translated audio, then blend the edited region back into its original shot.
4. **Fit the dub to the scene.** Remix the background sound and either preserve the original runtime or retime eligible segments together with their audio to match the translated speech.

<details>
<summary>Timing modes and voice options</summary>

**Fixed duration** fits translated speech into the original timeline. **Adaptive duration** can extend or shorten isolated dialogue segments within a single shot, adjusting video and audio together; segments that cross cuts or overlap other dialogue retain their original duration.

Local Qwen3-TTS supports voice cloning when suitable speaker references are available; API TTS uses the provider's preset voices. Dialogue without a reliable visible face is dubbed without a mouth edit.

</details>

<a id="quick-start"></a>
## Quick start

Requires **Linux, CUDA GPUs, FFmpeg, `curl`, `unzip`, and `uv`**. The example below uses local translation and TTS. **Translation can also use DeepSeek, OpenAI GPT, or another OpenAI-compatible API**—no local LightLLM/vLLM environment, Qwen3.5 weights, or translation GPU is needed. See [API setup and examples](docs/installation.md#api-backends).

### 1. Install

```bash
git clone https://github.com/ModelTC/LightDubber.git
cd LightDubber
bash scripts/install.sh --translation lightllm
```

The installer prepares the pipeline and local TTS environments; `--translation lightllm` also installs the local translation server. Local Qwen3-TTS currently uses a separate Python environment because its Transformers dependency differs from the pipeline's. You do not need to manage that environment manually. API TTS can skip it with `--tts api`.

Lip-sync requires a [LightX2V](https://github.com/ModelTC/LightX2V) installation with native TBDub support. Follow the [model-service setup](docs/installation.md#model-services) before launching jobs.

### 2. Download and configure

```bash
export MODEL_DIR=/path/to/models/video_dubbing
PYTHON_BIN="$PWD/.venv/bin/python" bash scripts/download_models.sh all
.venv/bin/python scripts/configure.py --models "$MODEL_DIR" --output config.local.json
```

[config.example.json](config.example.json) lists model paths and backend options. The generated local configuration is ignored by Git. See [model downloads](docs/installation.md#model-downloads) for individual components and download sources.

### 3. Start the model services

Start [LightLLM and LightX2V](docs/installation.md#model-services) in separate terminals. The documented example uses GPU 0 for translation, GPU 1 for lip-sync, and GPUs 2–3 for pipeline workers. Configure the GPU IDs and model paths for your machine.

| Service | Default address |
| :-- | :-- |
| Translation · LightLLM | `http://127.0.0.1:18440/v1` |
| Lip-sync · LightX2V | `http://127.0.0.1:18441` |

### 4. Translate a video

Start the web UI after the model services are ready:

```bash
DUB_GPUS=2,3 DUB_CONFIG="$PWD/config.local.json" \
  DUB_DATA_DIR="$PWD/outputs/web" bash scripts/serve.sh
```

Open **http://127.0.0.1:18320**, wait for worker model loading, then upload a video and choose the target language. Use `DUB_GPUS=2` for a single pipeline worker.

Or run the CLI on an idle pipeline GPU:

```bash
.venv/bin/python main.py \
  --config config.local.json --gpu 2 \
  --source input.mp4 --target-language en \
  --output outputs/example_en
```

See [usage options](docs/installation.md#usage-options) for duration modes, uploads, public access, and resuming jobs.

## Results and development

Each job writes `final.mp4`, `source.txt`, `translated.txt`, `report.json`, and execution logs. The report includes separate translation and TTS timings. Web jobs are saved under `$DUB_DATA_DIR/jobs/<id>/`.

<details>
<summary>Code layout and stage interfaces</summary>

```text
main.py                  CLI entry point
runtime/                 Pipeline, configuration, clients, resident processes
stages/
  prepare/               Normalize video and audio
  recognition/           ASR, alignment, acoustic speaker labels
  speakers/              Face detection, active speaking, identity binding
  separation/            Dialogue/background separation
  speech/                Translation and voice synthesis
  lipsync/               Crop planning and LightX2V requests
  postprocess/           Mix, composite, retime, export
web/                     Uploads, job queue, result viewer
scripts/                 Installation, downloads, service launchers
```

The CLI and web workers share `Pipeline`. Model stages expose `run_stage(...)` in `run.py`, model/service adapters in `model.py`, and helpers in `utils.py`. Export stages keep focused audio and video modules. `runtime/paths.py` owns job artifact paths. S3FD, LoCoNet, and SAM-Audio inference code lives with the stages that use it.

TTS reuses audio for identical requests and stops shortening retries that no longer improve timing. Face detection caches fixed-shape prior boxes and overlaps frame encoding with detection. CPU regression tests cover caching, stage dispatch, and resume:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

</details>

## Acknowledgements

Built with [LightX2V](https://github.com/ModelTC/LightX2V), [LightLLM](https://github.com/ModelTC/LightLLM), and the upstream models listed in [SOURCES.json](SOURCES.json). Original licenses and notices are preserved in [licenses/](licenses/). [vLLM](https://github.com/vllm-project/vllm) is available as an alternative local translation backend.
