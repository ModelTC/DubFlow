"use strict";

const $ = (id) => document.getElementById(id);
const languageNames = { auto: "自动检测" };
const durationModeNames = { fixed: "保留原片时长", adaptive_video: "视频时长自适应" };
const statusNames = { queued: "排队中", running: "正在译制", succeeded: "已完成", failed: "处理失败", canceled: "已取消" };
const stageGroups = [
  ["prepare"],
  ["vision", "identity", "audio_nemotron3", "fusion"],
  ["source_separation"],
  ["speech"],
  ["audio_mix", "target", "rois"],
  ["render", "compose", "compose_audio_only"],
  ["adaptive_duration", "report"],
];
const state = { csrf: "", limits: null, service: null, profiles: [], durationModes: ["fixed"], profileSelected: false, file: null, duration: 0, previewURL: "", jobs: [], selectedId: "", uploading: false, canceledUpload: false, uploadId: "", uploadXhr: null, resultKey: "", resultMode: "final", toastTimer: null, pollTimer: null, ready: false, refreshing: false };

function jobStatusName(job) {
  if (job.status !== "succeeded") return statusNames[job.status];
  if (job.translation_skipped) return "未翻译 · 已返回原片";
  if (job.stats.audio_only && !job.stats.lip_edited && !job.stats.partial_lip_edited) return "仅译音完成 · 未改口型";
  if (job.stats.audio_only || job.stats.partial_lip_edited) return "已完成 · 部分口型未编辑";
  return statusNames[job.status];
}

function renderProfile() {
  const profile = state.profiles.find((item) => item.id === $("modelProfile").value);
  $("profileModels").replaceChildren(...(profile?.models || []).map((model) => {
    const item = document.createElement("span");
    const stage = document.createElement("small");
    stage.textContent = model.stage;
    item.append(stage, model.name);
    return item;
  }));
  $("profileHint").textContent = profile && !profile.available ? profile.reason : "Qwen 识别并对齐完整音频；SAM-Audio Large-TV 分离对白与背景；翻译服务生成译文，Qwen TTS 配音，LightX2V TBDub Student 编辑口型。没有可绑定的嘴部时只替换译音，模型执行异常则报错，不将失败当作成功。不做反复内容核验；识别可能漏句，背景可能残留原对白。";
  $("profileHint").classList.toggle("unavailable", Boolean(profile && !profile.available));
}

function renderDuration() {
  $("durationHint").textContent = !state.durationModes.includes("adaptive_video")
    ? "当前服务尚未更新，暂只支持保留原片时长。"
    : $("durationMode").checked
    ? "开启：根据译音长短局部调整画面时长。成片可能变长或变短，运动和背景音节奏可能变化。"
    : "关闭：保留原片时长；译音过长时先缩译，仍放不下则调速。";
}

function sourceLanguageName(job) {
  if (job.source_language !== "auto") return languageNames[job.source_language] || job.source_language;
  const detected = languageNames[job.detected_source_language];
  return detected ? `自动检测（${detected}）` : languageNames.auto;
}

function toast(message, kind = "info") {
  clearTimeout(state.toastTimer);
  $("toastText").textContent = message;
  $("toast").dataset.kind = kind;
  $("toast").hidden = false;
  if (kind !== "error") state.toastTimer = setTimeout(() => { $("toast").hidden = true; }, 6500);
}

async function api(path, options = {}) {
  const { timeoutMs = 0, ...requestOptions } = options;
  const controller = timeoutMs ? new AbortController() : null;
  const timer = controller ? setTimeout(() => controller.abort(), timeoutMs) : null;
  try {
    const response = await fetch(path, {
      credentials: "same-origin",
      ...requestOptions,
      ...(controller ? { signal: controller.signal } : {}),
      headers: { ...(requestOptions.body ? { "Content-Type": "application/json" } : {}), ...(requestOptions.method ? { "X-CSRF-Token": state.csrf } : {}), ...requestOptions.headers },
    });
    let result;
    try { result = await response.json(); }
    catch { throw Object.assign(new Error(`服务入口返回非 JSON 响应（HTTP ${response.status}），请稍后重试。`), { status: response.status, retryable: true }); }
    if (!response.ok) throw Object.assign(new Error(typeof result.detail === "string" ? result.detail : `请求未完成（HTTP ${response.status}），请稍后重试。`), { status: response.status });
    return result;
  } catch (error) {
    if (error.name === "AbortError") throw Object.assign(new Error("请求超时，请检查网络后重试。"), { retryable: true });
    throw error;
  } finally {
    if (timer) clearTimeout(timer);
  }
}

function formatBytes(bytes) {
  return bytes >= 1024 ** 3 ? `${(bytes / 1024 ** 3).toFixed(1)} GB` : `${(bytes / 1024 ** 2).toFixed(1)} MB`;
}

function formatDuration(seconds) {
  return `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;
}

function formatDate(timestamp) {
  const date = new Date(typeof timestamp === "number" ? timestamp * 1000 : timestamp);
  return date.toLocaleString("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hour12: false });
}

function setOnline(online, service) {
  if (service) state.service = service;
  const warming = state.service && state.service.ready_workers < state.service.total_workers;
  $("serviceStatus").dataset.state = online ? warming ? "warming" : "ready" : "offline";
  $("serviceText").textContent = !online ? "连接暂时中断" : state.service ? `${state.service.total_workers} 卡常驻 · 就绪 ${state.service.ready_workers}/${state.service.total_workers} · 处理中 ${state.service.busy_workers}` : "服务在线";
}

function updateForm() {
  const tooLong = state.limits?.max_duration_seconds > 0 && state.duration > state.limits.max_duration_seconds;
  const profile = state.profiles.find((item) => item.id === $("modelProfile").value);
  const unavailable = !profile?.available;
  const warming = state.service && state.service.ready_workers === 0;
  $("targetSymbol").textContent = $("targetLanguage").value === "zh" ? "中" : $("targetLanguage").value.toUpperCase();
  $("submitButton").disabled = !state.ready || !state.file || state.uploading || tooLong || unavailable || warming;
  $("submitButton").querySelector(".button-spinner").hidden = !state.uploading;
  $("submitButton").querySelector(".submit-arrow").hidden = state.uploading;
  $("submitText").textContent = state.uploading ? "正在提交素材" : state.uploadId ? "继续提交" : "开始翻译";
  for (const id of ["targetLanguage", "modelProfile", "durationMode", "chooseFile", "replaceFile", "removeFile"]) $(id).disabled = state.uploading || id === "durationMode" && !state.durationModes.includes("adaptive_video");
  $("submitHint").classList.toggle("error", Boolean(tooLong));
  $("submitHint").textContent = !state.ready ? "正在连接本机翻译服务…" : unavailable ? profile?.reason || "模型配置尚未就绪。" : warming ? `模型预热中（${state.service.ready_workers}/${state.service.total_workers}），可以先选择视频。` : tooLong ? `视频超过 ${state.limits.max_duration_seconds} 秒，请先截取需要翻译的部分。` : state.uploading ? "素材分块上传，请保持页面开启。" : state.file ? `提交后分配给空闲显卡，最多 ${state.service.total_workers} 个视频同时处理。` : "选择一个视频，开始你的第一次翻译。";
}

function setFile(file) {
  if (state.uploading) return;
  if (!state.ready) return toast("服务尚未连接，请稍后再选择视频。", "error");
  if (file.size > state.limits.max_upload_bytes) return toast(`文件太大，当前上限为 ${formatBytes(state.limits.max_upload_bytes)}。`, "error");
  if (!file.size) return toast("这个文件为空，请选择有效的视频。", "error");
  if (!/\.(mp4|mov|mkv|webm|avi|m4v)$/i.test(file.name)) return toast("请选择 MP4、MOV、MKV 或 WebM 视频文件。", "error");
  discardUpload();
  if (state.previewURL) URL.revokeObjectURL(state.previewURL);
  state.file = file;
  state.duration = 0;
  state.previewURL = URL.createObjectURL(file);
  $("sourceVideo").src = state.previewURL;
  $("fileName").textContent = file.name;
  $("fileMeta").textContent = `${formatBytes(file.size)} · 正在读取时长`;
  $("uploadEmpty").hidden = true;
  $("videoPreview").hidden = false;
  $("sourceDetails").hidden = false;
  $("uploadProgress").hidden = true;
  updateForm();
}

function clearFile() {
  if (state.uploading) return;
  discardUpload();
  if (state.previewURL) URL.revokeObjectURL(state.previewURL);
  state.file = null;
  state.duration = 0;
  state.previewURL = "";
  $("sourceVideo").pause();
  $("sourceVideo").removeAttribute("src");
  $("sourceVideo").load();
  $("fileInput").value = "";
  $("uploadEmpty").hidden = false;
  $("videoPreview").hidden = true;
  $("sourceDetails").hidden = true;
  $("uploadProgress").hidden = true;
  updateForm();
}

function discardUpload() {
  const uploadId = state.uploadId;
  state.uploadId = "";
  if (uploadId) api(`/api/uploads/${uploadId}`, { method: "DELETE" }).catch(() => {});
}

function renderUploadProgress(confirmed, sent = null, chunkSize = 0) {
  const bytes = Math.min(confirmed, state.file.size);
  const percent = Math.round(100 * bytes / state.file.size);
  $("uploadProgressBar").value = percent;
  const progress = `服务器已确认 ${percent}% · ${formatBytes(bytes)} / ${formatBytes(state.file.size)}`;
  $("uploadProgressLabel").textContent = sent === null ? progress : sent >= chunkSize
    ? `${progress} · 当前分块已发送到隧道，等待服务器确认…`
    : `${progress} · 当前分块已发送 ${formatBytes(sent)} / ${formatBytes(chunkSize)} 到隧道`;
}

function sendChunk(uploadId, offset, chunk) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    let idleTimer;
    let stalled = false;
    const clearIdleTimer = () => clearTimeout(idleTimer);
    const resetIdleTimer = () => {
      clearIdleTimer();
      idleTimer = setTimeout(() => {
        stalled = true;
        xhr.abort();
        reject(Object.assign(new Error("上传通道 45 秒无进展，正在检查已接收的分块。"), { retryable: true }));
      }, 45000);
    };
    state.uploadXhr = xhr;
    xhr.open("PUT", `/api/uploads/${uploadId}`);
    xhr.setRequestHeader("X-CSRF-Token", state.csrf);
    xhr.setRequestHeader("Upload-Offset", String(offset));
    xhr.setRequestHeader("Content-Type", "application/octet-stream");
    xhr.timeout = 120000;
    xhr.upload.onprogress = (event) => {
      renderUploadProgress(offset, event.loaded, chunk.size);
      resetIdleTimer();
    };
    xhr.upload.onload = () => {
      renderUploadProgress(offset, chunk.size, chunk.size);
      resetIdleTimer();
    };
    xhr.onload = () => {
      clearIdleTimer();
      let data;
      const retryable = [408, 409, 429, 499].includes(xhr.status) || xhr.status >= 500;
      try { data = JSON.parse(xhr.responseText); }
      catch { return reject(Object.assign(new Error(`上传入口返回非 JSON 响应（HTTP ${xhr.status}），已提交分块仍保留。`), { retryable: retryable || xhr.status === 404, status: xhr.status })); }
      if ((xhr.status >= 200 && xhr.status < 300 && data.offset === offset + chunk.size) ||
          (xhr.status === 499 && data.offset > offset && data.offset <= offset + chunk.size)) {
        renderUploadProgress(data.offset);
        resolve(data.offset);
      } else {
        reject(Object.assign(new Error(typeof data.detail === "string" ? data.detail : `上传响应异常（HTTP ${xhr.status}），请重试。`), { retryable, status: xhr.status }));
      }
    };
    xhr.onerror = () => {
      clearIdleTimer();
      reject(Object.assign(new Error("上传连接中断，正在检查已接收的分块。"), { retryable: true }));
    };
    xhr.ontimeout = () => {
      clearIdleTimer();
      reject(Object.assign(new Error("分块上传超过 2 分钟，正在检查已接收的分块。"), { retryable: true }));
    };
    xhr.onabort = () => {
      clearIdleTimer();
      reject(stalled ? Object.assign(new Error("上传通道 45 秒无进展。"), { retryable: true }) : new Error("上传已取消。"));
    };
    renderUploadProgress(offset, 0, chunk.size);
    resetIdleTimer();
    xhr.send(chunk);
  });
}

async function uploadChunk(uploadId, offset, chunk) {
  for (let attempt = 0; attempt < 3; attempt++) {
    if (state.canceledUpload) throw new Error("上传已取消。");
    try { return await sendChunk(uploadId, offset, chunk); }
    catch (error) {
      if (state.canceledUpload || !error.retryable) throw error;
      // A lost response is not proof the chunk was lost. Check before resending.
      let remote;
      try { remote = await api(`/api/uploads/${uploadId}`, { timeoutMs: 15000 }); }
      catch (probeError) {
        if (attempt === 2 || probeError.status && probeError.status < 500 && !probeError.retryable) throw probeError;
        $("uploadProgressLabel").textContent = `连接暂时中断，正在重试当前分块（${attempt + 1}/2）…`;
        await new Promise(resolve => setTimeout(resolve, 1000 * (attempt + 1)));
        continue;
      }
      if (remote.status !== "uploading" || remote.offset < offset || remote.offset > offset + chunk.size) {
        throw new Error("上传状态发生变化，请刷新后查看任务记录。");
      }
      if (remote.offset > offset) {
        renderUploadProgress(remote.offset);
        return remote.offset;
      }
      if (attempt === 2) throw error;
      $("uploadProgressLabel").textContent = `连接中断，正在续传（重试 ${attempt + 1}/2）…`;
      await new Promise(resolve => setTimeout(resolve, 1000 * (attempt + 1)));
    }
  }
}

async function submit(event) {
  event.preventDefault();
  if ($("submitButton").disabled) return;
  state.uploading = true;
  state.canceledUpload = false;
  $("cancelUpload").disabled = false;
  $("uploadProgress").hidden = false;
  $("uploadProgressBar").value = 0;
  $("uploadProgressLabel").textContent = "准备上传…";
  updateForm();
  let submittingJob = false;
  try {
    let upload;
    if (state.uploadId) {
      upload = await api(`/api/uploads/${state.uploadId}`);
      if (upload.status === "submitted") {
        state.uploadId = "";
        await refreshJobs();
        toast("视频已经提交，请查看下方任务记录。");
        $("uploadProgress").hidden = true;
        return;
      }
      if (upload.status !== "uploading") {
        state.uploadId = "";
        throw new Error("上次上传已过期或取消，请重新提交。");
      }
      upload.chunk_bytes = state.limits.chunk_bytes;
    } else {
      upload = await api("/api/uploads", { method: "POST", body: JSON.stringify({ filename: state.file.name, size: state.file.size }) });
    }
    state.uploadId = upload.upload_id;
    let offset = upload.offset;
    renderUploadProgress(offset);
    while (offset < state.file.size) {
      if (state.canceledUpload) throw new Error("上传已取消。");
      const chunk = state.file.slice(offset, offset + upload.chunk_bytes);
      offset = await uploadChunk(upload.upload_id, offset, chunk);
    }
    if (state.canceledUpload) throw new Error("上传已取消。");
    $("uploadProgressLabel").textContent = "上传完成，正在检查视频并加入队列…";
    $("cancelUpload").disabled = true;
    submittingJob = true;
    const request = { upload_id: upload.upload_id, source_language: "auto", target_language: $("targetLanguage").value, profile: $("modelProfile").value };
    if (state.durationModes.includes("adaptive_video")) request.duration_mode = $("durationMode").checked ? "adaptive_video" : "fixed";
    const job = await api("/api/jobs", { method: "POST", body: JSON.stringify(request) });
    submittingJob = false;
    state.uploadId = "";
    state.selectedId = job.id;
    state.jobs.unshift(job);
    renderJobs();
    $("uploadProgress").hidden = true;
    toast("视频已加入译制队列，可以在下方查看处理进度。");
    $("tasksTitle").scrollIntoView({ behavior: "smooth", block: "start" });
    refreshJobs();
  } catch (error) {
    if (!state.canceledUpload) toast(`${error.message}${submittingJob ? " 请先检查下方译制记录；如果任务已经出现，不要重复提交。" : state.uploadId ? " 保持当前文件，点击“继续提交”可续传。" : ""}`, "error");
    if (submittingJob) await refreshJobs();
    if (state.canceledUpload) discardUpload();
    $("uploadProgress").hidden = true;
  } finally {
    state.uploading = false;
    state.uploadXhr = null;
    updateForm();
  }
}

function renderJobs() {
  $("jobCount").textContent = String(state.jobs.length).padStart(2, "0");
  $("tasksEmpty").hidden = state.jobs.length > 0;
  $("jobsWorkspace").hidden = state.jobs.length === 0;
  if (!state.jobs.length) return;
  if (!state.jobs.some((job) => job.id === state.selectedId)) state.selectedId = state.jobs[0].id;
  const activeCount = state.jobs.filter((job) => ["queued", "running"].includes(job.status)).length;
  $("activeJobCount").textContent = activeCount ? `${activeCount} 个处理中` : "";
  const focusedJob = document.activeElement.closest?.(".job-row")?.dataset.id;
  const scrollLeft = $("jobList").scrollLeft;
  $("jobList").replaceChildren(...state.jobs.map((job) => {
    const row = $("jobRowTemplate").content.firstElementChild.cloneNode(true);
    row.dataset.id = job.id;
    row.setAttribute("aria-current", String(job.id === state.selectedId));
    row.querySelector("strong").textContent = job.filename;
    row.querySelector(".job-row-meta").textContent = `${sourceLanguageName(job)} → ${languageNames[job.target_language]} · ${durationModeNames[job.duration_mode] || durationModeNames.fixed} · ${jobStatusName(job)}`;
    row.querySelector(".job-row-state").dataset.state = job.status;
    row.addEventListener("click", () => { state.selectedId = job.id; state.resultMode = "final"; renderJobs(); });
    return row;
  }));
  $("jobList").scrollLeft = scrollLeft;
  if (focusedJob) [...$("jobList").children].find((row) => row.dataset.id === focusedJob)?.focus({ preventScroll: true });
  renderJob(state.jobs.find((job) => job.id === state.selectedId));
}

function renderJob(job) {
  const running = job.status === "running";
  const queued = job.status === "queued";
  const succeeded = job.status === "succeeded";
  $("jobLanguage").textContent = `${sourceLanguageName(job)} → ${languageNames[job.target_language]}`;
  $("jobTitle").textContent = job.filename;
  $("jobDate").textContent = `${formatDate(job.created_at)}${job.duration ? ` · 原片 ${formatDuration(job.duration)}` : ""}${job.output_duration ? ` · 成片 ${formatDuration(job.output_duration)}` : ""} · ${durationModeNames[job.duration_mode] || durationModeNames.fixed}`;
  $("jobProfileLabel").textContent = job.profile_label;
  $("jobProfileLabel").dataset.profile = job.profile;
  $("jobModels").textContent = job.models.map((model) => model.name).join(" / ");
  $("jobStatus").textContent = jobStatusName(job);
  $("jobStatus").dataset.state = job.status;
  $("jobShortId").textContent = `TASK / ${job.id.slice(0, 10).toUpperCase()}`;
  $("cancelJob").hidden = !running && !queued;
  $("jobProcessing").hidden = !running && !queued;
  $("processingSpinner").hidden = !running;
  $("resultArea").hidden = !succeeded;
  $("jobError").hidden = !["failed", "canceled"].includes(job.status);
  $("jobError").textContent = job.status === "canceled" ? "这次译制已取消。你可以重新提交视频，开始新的任务。" : job.error || "处理未完成，请重新提交或联系服务维护者。";
  if (running || queued) {
    $("stageTitle").textContent = queued ? (job.queue_position ? `排队第 ${job.queue_position} 位` : "正在等待处理") : job.stage_label;
    $("stageDescription").textContent = queued ? "素材已准备好，轮到这个任务时将自动开始。" : "模型正在本机处理，完成后会自动显示预览和下载。";
    $("jobProgressBar").style.width = `${queued ? 0 : job.progress}%`;
    $("jobProgressCaption").textContent = queued ? "等待 GPU 空闲" : "按处理阶段更新，不代表剩余时间";
    $("jobProgressValue").textContent = queued ? "" : `阶段进度 ${job.progress}%`;
    const group = queued ? -1 : stageGroups.findIndex((stages) => stages.includes(job.stage));
    [...$("stageList").children].forEach((item, index) => {
      item.classList.toggle("active", index === group);
      item.classList.toggle("completed", index < group);
    });
  }
  if (succeeded) renderResult(job);
  else { $("resultVideo").pause(); state.resultKey = ""; }
}

function renderResult(job) {
  const files = job.files;
  const resultVideo = job.result_video_file || "final.mp4";
  $("comparisonTab").hidden = !files["comparison.mp4"];
  $("originalTab").hidden = !files.source;
  if (state.resultMode === "comparison" && !files["comparison.mp4"] || state.resultMode === "original" && !files.source) state.resultMode = "final";
  for (const [id, mode] of [["finalTab", "final"], ["comparisonTab", "comparison"], ["originalTab", "original"]]) {
    $(id).setAttribute("aria-selected", String(state.resultMode === mode));
    $(id).tabIndex = state.resultMode === mode ? 0 : -1;
  }
  const url = state.resultMode === "original" ? files.source : state.resultMode === "comparison" ? files["comparison.mp4"] : files[resultVideo];
  const key = `${job.id}:${state.resultMode}:${url}`;
  if (key !== state.resultKey) {
    $("resultVideo").pause();
    $("resultVideo").src = url;
    state.resultKey = key;
  }
  const completeness = job.audio_completeness;
  const stats = job.translation_skipped ? [] : completeness ? [[completeness.recognized_dialogue, "句已识别可译对白"], [completeness.replaced_dialogue, "句已替换对白"]] : [[job.stats.translated, "句已翻译"]];
  if (!job.translation_skipped) {
    stats.push([job.stats.lip_edited, "句口型已编辑"]);
    if (job.stats.partial_lip_edited) stats.push([job.stats.partial_lip_edited, "句口型部分编辑"]);
    stats.push([job.stats.audio_only, "句仅替换音频"], [completeness ? completeness.unreplaced_ids.length : job.stats.preserved_original, completeness ? "句未替换对白" : "句保留原声"]);
  }
  $("resultStats").replaceChildren(...stats.map(([count, label]) => {
    const span = document.createElement("span");
    const value = document.createElement("strong");
    value.textContent = count || 0;
    span.append(value, label);
    return span;
  }));
  $("resultDisclaimer").textContent = job.translation_skipped
    ? `自动识别的主要语言已经是目标语言，本次未翻译或修改视频，已返回原片。${job.other_detected_languages?.length ? "识别到其他语言片段，也未进行逐句翻译。" : ""}`
    : completeness ? `${completeness.recognized_dialogue === 0 ? "未检测到可译对白，不代表原片没有对白。" : completeness.unreplaced_ids.length ? "仍有已识别对白未替换，可在报告中查看。" : "已替换全部已识别可译对白。"}音乐、环境声和动物叫声等非对白不翻译。识别仍可能漏检，不能保证绝对无遗漏；请完整预览结果。` : "自动处理结果可能存在漏译、口型或音色偏差。建议下载前完整预览；保留原声的片段可在处理报告中查看。";
  if (job.stats.audio_only || job.stats.partial_lip_edited) $("resultDisclaimer").textContent = "部分或全部对白只替换了音频，未完成对应画面的口型编辑；详见统计和处理报告。 " + $("resultDisclaimer").textContent;
  if (!job.translation_skipped && job.content_verification === "disabled") $("resultDisclaimer").textContent += " 未做 ASR 内容核验和成片语言核验；配音放入时间窗与音轨替换成功不代表台词内容已经验证。背景可能残留原语言对白。";
  if (!job.translation_skipped && job.duration_mode === "adaptive_video") $("resultDisclaimer").textContent += " 此任务会按译音调整画面时长，局部运动和背景音节奏可能变化；请检查接缝与口型。";
  $("resultDownloads").replaceChildren(...[[resultVideo, "下载视频"], ["comparison.mp4", "对比视频"], ["source.txt", "原文"], ["translated.txt", "译文"], ["report.json", "处理报告"]].filter(([file]) => files[file]).map(([file, label], index) => {
    const link = document.createElement("a");
    link.href = files[file];
    link.download = job.translation_skipped && file === resultVideo ? job.filename : `${job.filename.replace(/\.[^.]+$/, "")}_${file}`;
    if (index === 0) link.className = "primary";
    const icon = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    icon.setAttribute("class", "icon");
    icon.setAttribute("aria-hidden", "true");
    const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
    use.setAttribute("href", "#i-download");
    icon.append(use);
    link.append(icon, label);
    return link;
  }));
}

async function refreshJobs() {
  if (state.refreshing) return;
  state.refreshing = true;
  clearTimeout(state.pollTimer);
  try {
    const data = await api("/api/jobs");
    state.jobs = data.jobs;
    renderJobs();
    setOnline(true, data.service);
    updateForm();
  } catch {
    setOnline(false);
  } finally {
    state.refreshing = false;
    state.pollTimer = setTimeout(refreshJobs, document.hidden ? 12000 : 3500);
  }
}

async function bootstrap() {
  try {
    const data = await api("/api/bootstrap");
    state.csrf = data.csrf_token;
    state.limits = data.limits;
    state.profiles = data.profiles;
    Object.assign(languageNames, Object.fromEntries(data.languages.map(({ code, label }) => [code, label])));
    const target = $("targetLanguage").value || "en";
    $("targetLanguage").replaceChildren(...data.languages.map(({ code, label }) => {
      const option = document.createElement("option");
      option.value = code;
      option.textContent = label;
      return option;
    }));
    $("targetLanguage").value = data.languages.some(({ code }) => code === target) ? target : "en";
    state.durationModes = data.duration_modes || ["fixed"];
    if (!state.durationModes.includes("adaptive_video")) $("durationMode").checked = false;
    const selected = state.profileSelected ? $("modelProfile").value : data.default_profile;
    $("modelProfile").replaceChildren(...data.profiles.map((profile) => {
      const option = document.createElement("option");
      option.value = profile.id;
      option.textContent = `${profile.label}${profile.available ? " · 当前路线" : " · 暂未就绪"}`;
      option.disabled = !profile.available;
      return option;
    }));
    $("modelProfile").value = selected;
    renderProfile();
    renderDuration();
    state.ready = true;
    const durationLimit = data.limits.max_duration_seconds > 0 ? `${data.limits.max_duration_seconds} 秒` : "不限制时长";
    $("uploadLimit").textContent = `MP4 / MOV / MKV / WebM · ${formatBytes(data.limits.max_upload_bytes)} · ${durationLimit} · 不预设分辨率上限`;
    setOnline(true, data.service);
    updateForm();
    await refreshJobs();
  } catch {
    setOnline(false);
    $("submitHint").textContent = "暂时无法连接服务，正在自动重连…";
    setTimeout(bootstrap, 7000);
  }
}

$("chooseFile").addEventListener("click", () => $("fileInput").click());
$("replaceFile").addEventListener("click", () => $("fileInput").click());
$("removeFile").addEventListener("click", clearFile);
$("fileInput").addEventListener("change", (event) => { if (event.target.files[0]) setFile(event.target.files[0]); });
$("sourceVideo").addEventListener("loadedmetadata", () => {
  if (!state.file) return;
  state.duration = Number.isFinite($("sourceVideo").duration) ? $("sourceVideo").duration : 0;
  $("fileMeta").textContent = `${formatBytes(state.file.size)}${state.duration ? ` · ${formatDuration(state.duration)}` : ""}${$("sourceVideo").videoWidth ? ` · ${$("sourceVideo").videoWidth} × ${$("sourceVideo").videoHeight}` : ""}`;
  updateForm();
});
$("sourceVideo").addEventListener("error", () => { if (state.file) $("fileMeta").textContent = `${formatBytes(state.file.size)} · 浏览器无法预览，提交后由服务器检查`; });
$("targetLanguage").addEventListener("change", updateForm);
$("modelProfile").addEventListener("change", () => { state.profileSelected = true; renderProfile(); updateForm(); });
$("durationMode").addEventListener("change", () => { renderDuration(); updateForm(); });
$("translateForm").addEventListener("submit", submit);
$("cancelUpload").addEventListener("click", () => { state.canceledUpload = true; state.uploadXhr?.abort(); });
$("cancelJob").addEventListener("click", async () => {
  $("cancelJob").disabled = true;
  try {
    await api(`/api/jobs/${state.selectedId}/cancel`, { method: "POST" });
    await refreshJobs();
    toast("已发送取消请求，正在停止当前任务。");
  } catch (error) { toast(error.message, "error"); }
  finally { $("cancelJob").disabled = false; }
});
for (const [id, mode] of [["finalTab", "final"], ["comparisonTab", "comparison"], ["originalTab", "original"]]) $(id).addEventListener("click", () => { state.resultMode = mode; renderJob(state.jobs.find((job) => job.id === state.selectedId)); });
document.querySelector(".result-tabs").addEventListener("keydown", (event) => {
  if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
  const tabs = [...document.querySelectorAll(".result-tabs button")].filter((button) => !button.hidden);
  const index = tabs.indexOf(document.activeElement);
  const next = event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1 : (index + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
  event.preventDefault();
  tabs[next].click();
  tabs[next].focus();
});
for (const button of document.querySelectorAll("[data-open-help]")) button.addEventListener("click", () => $("helpDialog").showModal());
for (const id of ["closeHelp", "confirmHelp"]) $(id).addEventListener("click", () => $("helpDialog").close());
$("helpDialog").addEventListener("click", (event) => { if (event.target === $("helpDialog")) { const box = $("helpDialog").getBoundingClientRect(); if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) $("helpDialog").close(); } });
$("dismissToast").addEventListener("click", () => { $("toast").hidden = true; });
let dragDepth = 0;
$("dropzone").addEventListener("dragenter", (event) => { event.preventDefault(); if (++dragDepth === 1 && !state.uploading) $("dropzone").classList.add("dragging"); });
$("dropzone").addEventListener("dragover", (event) => event.preventDefault());
$("dropzone").addEventListener("dragleave", () => { if (--dragDepth === 0) $("dropzone").classList.remove("dragging"); });
$("dropzone").addEventListener("drop", (event) => { event.preventDefault(); dragDepth = 0; $("dropzone").classList.remove("dragging"); if (event.dataTransfer.files.length) setFile(event.dataTransfer.files[0]); });
window.addEventListener("dragover", (event) => event.preventDefault());
window.addEventListener("drop", (event) => event.preventDefault());
document.addEventListener("visibilitychange", () => { if (!document.hidden && state.ready) refreshJobs(); });
window.addEventListener("beforeunload", (event) => { if (state.uploading) { event.preventDefault(); event.returnValue = ""; } });
bootstrap();
