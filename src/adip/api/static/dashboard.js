const $ = (id) => document.getElementById(id);
const DEEPSEEK_OPENAI_COMPATIBLE_ENDPOINT = "https://api.deepseek.com/chat/completions";

const state = {
  activeMode: "rag",
  lastResult: null,
  lastTitle: "Ready",
  profilesById: {},
  customProfiles: {},
  indexedDocuments: [],
};

document.addEventListener("DOMContentLoaded", () => {
  bindModeTabs();
  bindForms();
  refreshHealth();
  refreshBenchmark();
  refreshGenerationEval();
  refreshModelProfiles();
  refreshIndexedDocuments();
});

function bindModeTabs() {
  document.querySelectorAll(".mode-tab").forEach((button) => {
    button.addEventListener("click", () => {
      state.activeMode = button.dataset.mode;
      document.querySelectorAll(".mode-tab").forEach((item) => {
        item.classList.toggle("active", item === button);
      });
      document.querySelectorAll("[data-mode-panel]").forEach((panel) => {
        panel.classList.toggle("hidden", panel.dataset.modePanel !== state.activeMode);
      });
      if (state.activeMode === "history") {
        loadAgentHistory();
        loadMlopsHistory();
      }
    });
  });
}

function bindForms() {
  $("ragForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    await runRequest(event.currentTarget, "/rag/query", buildRagPayload(), renderRagResult);
  });

  $("agentForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    await runRequest(event.currentTarget, "/agent/run", buildAgentPayload(), renderAgentResult);
  });

  $("indexForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    await runRequest(event.currentTarget, "/pipeline/rebuild-index", buildIndexPayload(), renderIndexResult);
  });

  $("refreshAgentHistory").addEventListener("click", () => loadAgentHistory());
  $("refreshMlopsHistory").addEventListener("click", () => loadMlopsHistory());
  $("uploadDocument").addEventListener("click", () => uploadDocument(false));
  $("uploadAndRebuild").addEventListener("click", () => uploadDocument(true));
  $("exportReport").addEventListener("click", () => exportReport());
  $("agentApiKey").addEventListener("input", () => renderApiKeyStatus());
  $("agentEndpointUrl").addEventListener("input", () => renderApiKeyStatus());
  $("addCustomModel").addEventListener("click", () => addCustomModel());
  $("testCustomModel").addEventListener("click", () => testCustomModel());
  $("ragIndexPath").addEventListener("change", () => refreshIndexedDocuments());
  $("agentIndexPath").addEventListener("change", () => refreshIndexedDocuments());
  $("indexPath").addEventListener("change", () => refreshIndexedDocuments());
}

async function refreshHealth() {
  const status = $("serviceStatus");
  try {
    const payload = await getJson("/health");
    status.textContent = payload.status === "ok" ? "Online" : "Check";
    status.classList.toggle("ok", payload.status === "ok");
    status.classList.remove("error");
  } catch (error) {
    status.textContent = "Offline";
    status.classList.add("error");
    status.classList.remove("ok");
  }
}

async function refreshBenchmark() {
  try {
    const payload = await getJson("/monitoring/retrieval-benchmark");
    if (!payload.available) {
      return;
    }
    $("bestBackend").textContent = labelMetric(payload.best_backend_by_mrr);
    $("bestVariant").textContent = labelMetric(payload.best_variant_by_mrr);
    $("tfidfMrr").textContent = formatScore(payload.metrics.tfidf_mrr);
    $("crossMrr").textContent = formatScore(payload.metrics.tfidf_cross_encoder_rerank_mrr);
  } catch (error) {
    console.warn(error);
  }
}

async function refreshGenerationEval() {
  try {
    const payload = await getJson("/monitoring/generation-eval");
    if (!payload.available) {
      return;
    }
    $("genFaithfulness").textContent = formatScore(payload.faithfulness);
    $("genGroundedRate").textContent = formatPercent(payload.grounded_rate);
    $("genExpectedCoverage").textContent = formatPercent(payload.expected_coverage);
    $("genCitationCoverage").textContent = formatPercent(payload.citation_coverage);
  } catch (error) {
    console.warn(error);
  }
}

async function runRequest(form, path, payload, renderer) {
  const button = form.querySelector("button[type='submit']");
  const originalText = button.innerHTML;
  button.disabled = true;
  button.innerHTML = '<span class="button-icon">...</span> Running';
  setResultTitle("Running");

  try {
    const result = await postJson(path, payload);
    renderer(result);
  } catch (error) {
    renderError(error);
  } finally {
    button.disabled = false;
    button.innerHTML = originalText;
    refreshHealth();
  }
}

function buildRagPayload() {
  const documentFilter = $("ragDocumentFilter").value.trim();
  return {
    question: $("ragQuestion").value.trim(),
    index_path: $("ragIndexPath").value.trim(),
    document_filter: documentFilter || null,
    top_k: Number($("ragTopK").value),
    candidate_k: Number($("ragCandidateK").value),
    reranker: $("ragReranker").value,
    allow_reranker_download: $("allowRerankerDownload").checked,
  };
}

function buildAgentPayload() {
  const reasoningProfile = $("reasoningModelProfile").value.trim();
  const writer = selectedProfilePayload($("agentModelProfile").value, "writer");
  const reasoner = selectedProfilePayload(reasoningProfile, "reasoner");
  const endpointUrl = $("agentEndpointUrl").value.trim();
  const apiKey = $("agentApiKey").value.trim();
  const documentFilter = $("agentDocumentFilter").value.trim();
  return {
    question: $("agentQuestion").value.trim(),
    index_path: $("agentIndexPath").value.trim(),
    document_filter: documentFilter || null,
    task: $("agentTask").value,
    domain: $("agentDomain").value,
    top_k: Number($("agentTopK").value),
    llm_provider: writer.provider,
    model_profile: writer.modelProfile,
    model_name: writer.modelName,
    endpoint_url: writer.endpointUrl || endpointUrl || null,
    api_key: writer.apiKey || apiKey || null,
    device: $("agentDevice").value,
    max_new_tokens: Number($("agentMaxTokens").value),
    reasoning_effort: $("agentReasoningEffort").value,
    reasoning_provider: reasoner.provider,
    reasoning_model_profile: reasoner.modelProfile,
    reasoning_model_name: reasoner.modelName,
    reasoning_endpoint_url: reasoner.endpointUrl || endpointUrl || null,
    reasoning_api_key: reasoner.apiKey || apiKey || null,
    reasoning_device: $("reasoningDevice").value,
    reasoning_max_new_tokens: Number($("reasoningMaxTokens").value),
    use_reasoning_planner: $("useReasoningPlanner").checked,
  };
}

function selectedProfilePayload(profileId, slot) {
  if (!profileId) {
    return {
      provider: null,
      modelProfile: null,
      modelName: null,
      endpointUrl: null,
      apiKey: null,
    };
  }
  const profile = state.profilesById[profileId];
  if (profile && profile.session) {
    return {
      provider: "openai_compatible",
      modelProfile: null,
      modelName: profile.model_name,
      endpointUrl: profile.session.endpoint_url,
      apiKey: profile.session.api_key,
    };
  }
  return {
    provider: null,
    modelProfile: slot === "reasoner" ? profileId || null : profileId,
    modelName: null,
    endpointUrl: null,
    apiKey: null,
  };
}

async function refreshModelProfiles() {
  try {
    const payload = await getJson("/model-profiles");
    const profiles = payload.items || [];
    const allProfiles = mergeProfiles(profiles);
    state.profilesById = Object.fromEntries(allProfiles.map((profile) => [profile.profile_id, profile]));
    populateModelSelect($("agentModelProfile"), allProfiles, "extractive_baseline", false);
    populateModelSelect($("reasoningModelProfile"), allProfiles, "", true);
    renderModelButtons("writerModelButtons", "agentModelProfile", allProfiles, [
      "extractive_baseline",
      "qwen3_8b_default",
      "deepseek_v4_flash_cloud",
      "deepseek_v4_pro_cloud",
    ]);
    renderModelButtons("reasoningModelButtons", "reasoningModelProfile", allProfiles, [
      "",
      "deepseek_r1_distill_qwen_14b_reasoning",
      "deepseek_v4_pro_cloud",
      "extractive_baseline",
    ]);
    renderApiKeyStatus();
  } catch (error) {
    console.warn(error);
  }
}

async function refreshIndexedDocuments(preferredFilename) {
  const indexPath = $("indexPath").value.trim() || $("agentIndexPath").value.trim() || "data/processed/vector_index";
  try {
    const payload = await getJson(`/index/documents?index_path=${encodeURIComponent(indexPath)}`);
    state.indexedDocuments = payload.items || [];
    populateDocumentSelect($("ragDocumentFilter"), state.indexedDocuments, preferredFilename);
    populateDocumentSelect($("agentDocumentFilter"), state.indexedDocuments, preferredFilename);
  } catch (error) {
    state.indexedDocuments = [];
    populateDocumentSelect($("ragDocumentFilter"), [], null);
    populateDocumentSelect($("agentDocumentFilter"), [], null);
    console.warn(error);
  }
}

function populateDocumentSelect(select, documents, preferredFilename) {
  const current = select.value;
  select.innerHTML = "";
  select.appendChild(new Option("All Indexed Documents", ""));
  documents.forEach((documentItem) => {
    const label = documentLabel(documentItem);
    select.appendChild(new Option(label, documentItem.document_id));
  });
  const preferred = documents.find((documentItem) => documentItem.filename === preferredFilename);
  const documentIds = new Set(documents.map((documentItem) => documentItem.document_id));
  if (preferred) {
    select.value = preferred.document_id;
  } else if (documentIds.has(current)) {
    select.value = current;
  }
}

function documentLabel(documentItem) {
  const pages = Number(documentItem.page_count || 0);
  const chunks = Number(documentItem.chunk_count || 0);
  const pageText = pages ? `${pages}p` : "pages ?";
  return `${documentItem.filename || documentItem.document_id} (${pageText}, ${chunks} chunks)`;
}

function mergeProfiles(profiles) {
  return [...profiles, ...Object.values(state.customProfiles)];
}

function populateModelSelect(select, profiles, selectedValue, includeNone) {
  const current = select.value || selectedValue;
  select.innerHTML = "";
  if (includeNone) {
    select.appendChild(new Option("None", ""));
  }
  profiles.forEach((profile) => {
    select.appendChild(new Option(modelLabel(profile), profile.profile_id));
  });
  select.value = current;
}

function renderModelButtons(containerId, selectId, profiles, preferredIds) {
  const container = $(containerId);
  const select = $(selectId);
  const byId = Object.fromEntries(profiles.map((profile) => [profile.profile_id, profile]));
  container.innerHTML = "";
  preferredIds.forEach((profileId) => {
    if (profileId && !byId[profileId]) {
      return;
    }
    const button = document.createElement("button");
    button.type = "button";
    button.className = "model-chip";
    button.dataset.profileId = profileId;
    button.textContent = profileId ? modelShortLabel(byId[profileId]) : "No Reasoner";
    button.addEventListener("click", () => {
      select.value = profileId;
      syncModelButtons(container, select.value);
      renderApiKeyStatus();
    });
    container.appendChild(button);
  });
  select.addEventListener("change", () => {
    syncModelButtons(container, select.value);
    renderApiKeyStatus();
  });
  syncModelButtons(container, select.value);
}

function addCustomModel() {
  const status = $("customModelStatus");
  const values = customModelValues();
  if (!values.ok) {
    setFormStatus(status, values.error, "missing");
    renderError(new Error(values.error));
    return;
  }

  const profileId = `custom_${slugify(values.label)}_${Date.now()}`;
  const profile = {
    profile_id: profileId,
    display_name: values.label,
    description: "Session-only custom API model.",
    role: "custom_api",
    provider: "openai_compatible",
    model_name: values.modelName,
    context_window: 0,
    max_new_tokens: Number($("agentMaxTokens").value) || 4096,
    quantization: "provider_managed",
    local_files_only: false,
    recommended_for: ["dashboard session"],
    serving: {},
    runtime: {
      provider: "openai_compatible",
      endpoint_env: "session",
      endpoint_configured: true,
      api_key_env: "session",
      api_key_configured: true,
      uses_api_key: true,
    },
    session: {
      endpoint_url: values.endpointUrl,
      api_key: values.apiKey,
    },
  };
  state.customProfiles[profileId] = profile;
  refreshCustomProfileSelects(profileId);
  $("agentEndpointUrl").value = values.endpointUrl;
  $("agentApiKey").value = values.apiKey;
  $("customModelApiKey").value = "";
  setFormStatus(status, `${values.label} added to Writer Model and Reasoning Model.`, "ready");
  setResultTitle("Custom Model Added");
  setLatency(null);
  $("answerText").textContent = [
    `Model: ${values.label}`,
    `API name: ${values.modelName}`,
    `Endpoint: ${values.endpointUrl}`,
    "",
    "The API key is kept only in this browser session.",
  ].join("\n");
  renderQuality(null);
  renderCitations([]);
  renderTrace([]);
  renderRaw({
    status: "custom_model_added",
    profile_id: profileId,
    label: values.label,
    model_name: values.modelName,
    endpoint_url: values.endpointUrl,
    api_key: "***",
  });
}

async function testCustomModel() {
  const status = $("customModelStatus");
  const values = customModelValues();
  if (!values.ok) {
    setFormStatus(status, values.error, "missing");
    renderError(new Error(values.error));
    return;
  }
  setFormStatus(status, "Testing model connection...", "");
  try {
    const payload = await postJson("/models/check", {
      model_name: values.modelName,
      endpoint_url: values.endpointUrl,
      api_key: values.apiKey,
      max_new_tokens: 128,
    });
    if (payload.ok) {
      setFormStatus(status, `${values.label} test passed.`, "ready");
    } else {
      setFormStatus(status, payload.error || "Model test failed.", "missing");
    }
    setResultTitle("Model Test");
    setLatency(payload.latency_ms);
    $("answerText").textContent = payload.ok
      ? `Model test passed.\nPreview: ${payload.preview || "-"}`
      : `Model test failed.\n${payload.error || "-"}`;
    renderQuality(null);
    renderCitations([]);
    renderTrace([]);
    renderRaw({ ...payload, api_key: "***" });
  } catch (error) {
    setFormStatus(status, error.message, "missing");
    renderError(error);
  }
}

function customModelValues() {
  const rawLabel = $("customModelLabel").value.trim();
  const modelName = $("customModelName").value.trim();
  const endpointUrl =
    $("customModelEndpoint").value.trim() ||
    $("agentEndpointUrl").value.trim() ||
    defaultEndpointForModel(modelName);
  const apiKey = $("customModelApiKey").value.trim() || $("agentApiKey").value.trim();
  const label = rawLabel || modelName || "Custom API Model";
  if (!modelName) {
    return { ok: false, error: "Model API Name is required." };
  }
  if (!endpointUrl) {
    return {
      ok: false,
      error: "Model API Endpoint is required for non-DeepSeek providers. For Gemini use https://generativelanguage.googleapis.com/v1beta/openai/chat/completions.",
    };
  }
  if (!validEndpointUrl(endpointUrl)) {
    return {
      ok: false,
      error: "Model API Endpoint must contain only the URL. Do not paste labels like `Model API Key:` or the key into the endpoint field.",
    };
  }
  if (!apiKey) {
    return { ok: false, error: "Model API Key is required." };
  }
  return { ok: true, label, modelName, endpointUrl, apiKey };
}

function defaultEndpointForModel(modelName) {
  return modelName.toLowerCase().includes("deepseek") ? DEEPSEEK_OPENAI_COMPATIBLE_ENDPOINT : "";
}

function validEndpointUrl(value) {
  try {
    const url = new URL(value);
    return (url.protocol === "https:" || url.protocol === "http:") && !/\s/.test(value);
  } catch (error) {
    return false;
  }
}

function setFormStatus(node, message, mode) {
  if (!node) {
    return;
  }
  node.textContent = message;
  node.classList.toggle("ready", mode === "ready");
  node.classList.toggle("missing", mode === "missing");
}

function refreshCustomProfileSelects(selectedProfileId) {
  const allProfiles = Object.values(state.profilesById)
    .filter((profile) => !profile.session)
    .concat(Object.values(state.customProfiles));
  state.profilesById = Object.fromEntries(allProfiles.map((profile) => [profile.profile_id, profile]));
  populateModelSelect($("agentModelProfile"), allProfiles, selectedProfileId, false);
  populateModelSelect($("reasoningModelProfile"), allProfiles, $("reasoningModelProfile").value || "", true);
  renderModelButtons("writerModelButtons", "agentModelProfile", allProfiles, [
    "extractive_baseline",
    "qwen3_8b_default",
    "deepseek_v4_flash_cloud",
    "deepseek_v4_pro_cloud",
  ]);
  renderModelButtons("reasoningModelButtons", "reasoningModelProfile", allProfiles, [
    "",
    "deepseek_r1_distill_qwen_14b_reasoning",
    "deepseek_v4_pro_cloud",
    "extractive_baseline",
  ]);
  $("agentModelProfile").value = selectedProfileId;
  renderApiKeyStatus();
}

function syncModelButtons(container, selectedValue) {
  container.querySelectorAll(".model-chip").forEach((button) => {
    button.classList.toggle("active", button.dataset.profileId === selectedValue);
  });
}

function modelLabel(profile) {
  const role = profile.role ? ` (${profile.role})` : "";
  return `${modelShortLabel(profile)}${role}`;
}

function modelShortLabel(profile) {
  if (!profile) {
    return "Unknown";
  }
  if (profile.profile_id === "extractive_baseline") return "Extractive";
  if (profile.profile_id === "qwen3_8b_default") return "Qwen 8B";
  if (profile.profile_id === "deepseek_r1_distill_qwen_14b_reasoning") return "DeepSeek 14B";
  if (profile.profile_id === "deepseek_r1_distill_qwen_32b_stretch") return "DeepSeek 32B";
  if (profile.profile_id === "deepseek_v4_flash_cloud") return "DeepSeek API Fast";
  if (profile.profile_id === "deepseek_v4_pro_cloud") return "DeepSeek API Pro";
  if (profile.session) return profile.display_name || "Custom API Model";
  return profile.profile_id;
}

function renderApiKeyStatus() {
  const node = $("apiKeyStatus");
  if (!node) {
    return;
  }
  const writer = state.profilesById[$("agentModelProfile").value];
  const reasoner = state.profilesById[$("reasoningModelProfile").value];
  const statuses = [
    modelRuntimeStatus("Writer", writer),
    modelRuntimeStatus("Reasoner", reasoner),
  ].filter(Boolean);
  node.textContent = statuses.length ? statuses.join(" | ") : "Model runtime: local";
  node.classList.toggle("ready", statuses.every((item) => !item.includes("missing")));
  node.classList.toggle("missing", statuses.some((item) => item.includes("missing")));
}

function modelRuntimeStatus(label, profile) {
  if (!profile) {
    return "";
  }
  if (profile.provider !== "openai_compatible") {
    return `${label}: local`;
  }
  const runtime = profile.runtime || {};
  const endpointEntered = Boolean($("agentEndpointUrl").value.trim());
  const apiKeyEntered = Boolean($("agentApiKey").value.trim());
  if (!runtime.endpoint_configured && !endpointEntered) {
    return `${label}: missing endpoint`;
  }
  if (apiKeyEntered) {
    return `${label}: API key entered`;
  }
  if (runtime.api_key_configured) {
    return `${label}: API key ready`;
  }
  return `${label}: missing ${runtime.api_key_env || "API key"}`;
}

function buildIndexPayload() {
  return {
    input_path: $("inputPath").value.trim(),
    chunks_path: $("chunksPath").value.trim(),
    index_path: $("indexPath").value.trim(),
    backend: $("indexBackend").value,
    chunk_size: Number($("chunkSize").value),
    chunk_overlap: Number($("chunkOverlap").value),
  };
}

function renderRagResult(payload) {
  setResultTitle("RAG Query");
  setLatency(payload.latency_ms);
  renderAnswer(payload.answer || "");
  renderQuality(payload.quality);
  renderCitations(payload.retrieved || []);
  renderTrace([]);
  renderRaw(payload);
}

function renderAgentResult(payload) {
  const statePayload = payload.state || {};
  setResultTitle("Agent Run");
  setLatency(payload.latency_ms);
  renderAnswer(agentAnswerText(statePayload));
  renderAnswerWarning(payload.answer_warning);
  renderQuality(payload.quality);
  renderCitations(statePayload.retrieved || []);
  renderTrace(statePayload.trace || []);
  renderRaw(payload);
}

function agentAnswerText(statePayload) {
  if (statePayload.final_answer) {
    return statePayload.final_answer;
  }
  const notes = statePayload.verification_notes || [];
  if (statePayload.status === "failed") {
    return [
      "Run failed before a final answer was produced.",
      "",
      ...notes.map((note) => `- ${note}`),
      "",
      "Open Raw JSON or History for the full trace.",
    ].join("\n");
  }
  return "No final answer recorded.";
}

function renderIndexResult(payload) {
  setResultTitle("Index Rebuild");
  setLatency(payload.latency_ms);
  const index = payload.index || {};
  const ingestion = payload.ingestion || {};
  renderAnswer(
    [
      `Status: ${payload.status}`,
      `Documents: ${ingestion.document_count}`,
      `Chunks: ${ingestion.chunk_count}`,
      `Backend: ${index.backend}`,
      `Index: ${index.index_path}`,
    ].join("\n"),
  );
  renderQuality(null);
  renderCitations([]);
  renderTrace([]);
  renderRaw(payload);
  refreshBenchmark();
  refreshIndexedDocuments();
}

async function uploadDocument(rebuildAfterUpload) {
  const fileInput = $("documentUpload");
  if (!fileInput.files.length) {
    renderError(new Error("Choose a PDF, Markdown, or text file first."));
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);
  formData.append("raw_dir", $("inputPath").value.trim() || "data/raw");
  setResultTitle(rebuildAfterUpload ? "Uploading and Rebuilding" : "Uploading");
  setLatency(null);

  try {
    const response = await fetch("/documents/upload", {
      method: "POST",
      body: formData,
    });
    const uploadPayload = await parseResponse(response);
    if (!rebuildAfterUpload) {
      renderUploadResult(uploadPayload);
      return;
    }
    const rebuildPayload = await postJson("/pipeline/rebuild-index", buildIndexPayload());
    renderUploadRebuildResult(uploadPayload, rebuildPayload);
  } catch (error) {
    renderError(error);
  } finally {
    refreshHealth();
  }
}

function renderUploadResult(payload) {
  setResultTitle("Document Upload");
  setLatency(null);
  $("answerText").textContent = [
    `Status: ${payload.status}`,
    `File: ${payload.filename}`,
    `Path: ${payload.path}`,
    `Size: ${payload.size_bytes} bytes`,
    "",
    "Rebuild the index to make this document searchable.",
  ].join("\n");
  renderQuality(null);
  renderCitations([]);
  renderTrace([]);
  renderRaw(payload);
}

function renderUploadRebuildResult(uploadPayload, rebuildPayload) {
  setResultTitle("Upload + Index Rebuild");
  setLatency(rebuildPayload.latency_ms);
  const index = rebuildPayload.index || {};
  const ingestion = rebuildPayload.ingestion || {};
  $("answerText").textContent = [
    `Uploaded: ${uploadPayload.filename}`,
    `Path: ${uploadPayload.path}`,
    `Index status: ${rebuildPayload.status}`,
    `Documents: ${ingestion.document_count}`,
    `Chunks: ${ingestion.chunk_count}`,
    `Backend: ${index.backend}`,
    `Index: ${index.index_path}`,
    "",
    "The uploaded document is now searchable.",
  ].join("\n");
  renderQuality(null);
  renderCitations([]);
  renderTrace([]);
  renderRaw({
    upload: uploadPayload,
    rebuild: rebuildPayload,
  });
  refreshBenchmark();
  refreshIndexedDocuments(uploadPayload.filename);
}

async function loadAgentHistory() {
  const limit = Number($("historyLimit").value) || 12;
  setHistoryLoading("agentHistoryList");
  try {
    const payload = await getJson(`/history/agent-traces?limit=${encodeURIComponent(limit)}`);
    renderAgentHistory(payload.items || []);
  } catch (error) {
    renderHistoryError("agentHistoryList", error);
  }
}

async function loadMlopsHistory() {
  const limit = Number($("historyLimit").value) || 12;
  setHistoryLoading("mlopsHistoryList");
  try {
    const payload = await getJson(`/history/mlops-runs?limit=${encodeURIComponent(limit)}`);
    renderMlopsHistory(payload.items || []);
  } catch (error) {
    renderHistoryError("mlopsHistoryList", error);
  }
}

function renderAgentHistory(items) {
  const list = $("agentHistoryList");
  list.innerHTML = "";
  if (!items.length) {
    list.appendChild(emptyState("No agent traces"));
    return;
  }
  items.forEach((item) => {
    const node = document.createElement("button");
    node.type = "button";
    node.className = "history-item";
    node.innerHTML = `
      <p class="item-title">${escapeHtml(item.run_id)}</p>
      <p class="item-meta">${escapeHtml(item.status || "-")} | ${escapeHtml(item.task_type || "-")} | ${formatScore(item.workflow_duration_ms)} ms</p>
      <p class="item-meta">${escapeHtml(snippet(item.question || ""))}</p>
    `;
    node.addEventListener("click", () => loadAgentTraceDetail(item.run_id));
    list.appendChild(node);
  });
}

function renderMlopsHistory(items) {
  const list = $("mlopsHistoryList");
  list.innerHTML = "";
  if (!items.length) {
    list.appendChild(emptyState("No MLOps runs"));
    return;
  }
  items.forEach((item) => {
    const node = document.createElement("button");
    node.type = "button";
    node.className = "history-item";
    const bestVariant = item.key_params ? item.key_params.best_variant_by_mrr : null;
    node.innerHTML = `
      <p class="item-title">${escapeHtml(item.run_id)}</p>
      <p class="item-meta">${escapeHtml(item.status || "-")} | ${escapeHtml(item.run_name || "-")}</p>
      <p class="item-meta">${escapeHtml(bestVariant || `${item.metric_count || 0} metrics`)}</p>
    `;
    node.addEventListener("click", () => loadMlopsRunDetail(item.run_id));
    list.appendChild(node);
  });
}

async function loadAgentTraceDetail(runId) {
  try {
    const payload = await getJson(`/history/agent-traces/${encodeURIComponent(runId)}`);
    const trace = payload.trace || {};
    setResultTitle(`Trace ${runId}`);
    setLatency(trace.metrics ? trace.metrics.workflow_duration_ms : null);
    renderAnswer(trace.final_answer || "No final answer recorded.");
    renderAnswerWarning(trace.llmops ? trace.llmops.answer_warning : null);
    renderQuality(trace.llmops ? trace.llmops.quality : null);
    renderCitations(trace.retrieved || []);
    renderTrace(trace.trace || []);
    renderRaw(payload);
  } catch (error) {
    renderError(error);
  }
}

async function loadMlopsRunDetail(runId) {
  try {
    const payload = await getJson(`/history/mlops-runs/${encodeURIComponent(runId)}`);
    const run = payload.run || {};
    setResultTitle(`MLOps ${runId}`);
    setLatency(run.duration_ms);
    renderAnswer(summarizeMlopsRun(run));
    renderQuality(null);
    renderCitations([]);
    renderMlopsMetrics(run.metrics || {});
    renderRaw(payload);
  } catch (error) {
    renderError(error);
  }
}

function summarizeMlopsRun(run) {
  const params = run.params || {};
  const metrics = run.metrics || {};
  return [
    `Run: ${run.run_name || run.run_id}`,
    `Status: ${run.status || "-"}`,
    `Started: ${run.started_at || "-"}`,
    `Ended: ${run.ended_at || "-"}`,
    `Best backend: ${params.best_backend_by_mrr || "-"}`,
    `Best variant: ${params.best_variant_by_mrr || "-"}`,
    `MRR: ${formatScore(metrics.mrr || metrics.tfidf_cross_encoder_rerank_mrr || metrics.tfidf_mrr)}`,
    `Metrics logged: ${Object.keys(metrics).length}`,
    `Artifacts logged: ${Object.keys(run.artifacts || {}).length}`,
  ].join("\n");
}

function renderMlopsMetrics(metrics) {
  const events = Object.entries(metrics)
    .filter(([key]) => key.includes("mrr") || key.includes("latency") || key.includes("chunk_count"))
    .slice(0, 10)
    .map(([key, value]) => ({
      node_name: key,
      status: formatScore(value),
      duration_ms: null,
      output_summary: {},
    }));
  renderTrace(events);
}

function renderError(error) {
  setResultTitle("Request Error");
  setLatency(null);
  $("answerText").textContent = error.message;
  renderQuality(null);
  renderCitations([]);
  renderTrace([]);
  renderRaw({ error: error.message });
}

function renderQuality(quality) {
  const grid = $("qualityGrid");
  grid.innerHTML = "";
  if (!quality) {
    grid.appendChild(emptyState("No quality metrics"));
    return;
  }
  const items = [
    ["Fidelity", formatScore(quality.fidelity_score)],
    ["Coverage", formatPercent(quality.citation_coverage)],
    ["Visible Citations", safeDisplay(quality.visible_citation_count)],
    ["Unsupported", safeDisplay(quality.unsupported_sentence_count)],
    ["Claims", safeDisplay(quality.answer_sentence_count)],
    ["Evidence", safeDisplay(quality.evidence_count)],
  ];
  items.forEach(([label, value]) => {
    const node = document.createElement("div");
    node.className = "quality-item";
    node.innerHTML = `
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(value)}</strong>
    `;
    grid.appendChild(node);
  });
}

function renderCitations(items) {
  const list = $("citationList");
  list.innerHTML = "";
  if (!items.length) {
    list.appendChild(emptyState("No citations"));
    return;
  }
  items.forEach((item) => {
    const chunk = item.chunk || {};
    const node = document.createElement("article");
    node.className = "citation-item";
    node.innerHTML = `
      <p class="item-title">${escapeHtml(item.citation || chunk.filename || "Source")}</p>
      <p class="item-meta">Rank ${item.rank} | Score ${formatScore(item.score)} | ${escapeHtml(chunk.chunk_id || "")}</p>
      <p class="item-meta">${escapeHtml(snippet(chunk.text || ""))}</p>
    `;
    list.appendChild(node);
  });
}

function renderTrace(events) {
  const list = $("traceList");
  list.innerHTML = "";
  if (!events.length) {
    list.appendChild(emptyState("No trace events"));
    return;
  }
  events.forEach((event) => {
    const node = document.createElement("article");
    node.className = "trace-item";
    node.innerHTML = `
      <p class="item-title">${escapeHtml(event.node_name || "node")}</p>
      <p class="item-meta">${escapeHtml(event.status || "status")}${event.duration_ms == null ? "" : ` | ${formatScore(event.duration_ms)} ms`}</p>
      <p class="item-meta">Retrieved ${safeCount(event.output_summary, "retrieved_count")} | Citations ${safeCount(event.output_summary, "citation_count")}</p>
    `;
    list.appendChild(node);
  });
}

function setHistoryLoading(id) {
  const list = $(id);
  list.innerHTML = "";
  list.appendChild(emptyState("Loading"));
}

function renderHistoryError(id, error) {
  const list = $(id);
  list.innerHTML = "";
  list.appendChild(emptyState(error.message));
}

function emptyState(text) {
  const node = document.createElement("div");
  node.className = "empty-state";
  node.textContent = text;
  return node;
}

async function getJson(path) {
  const response = await fetch(path);
  return parseResponse(response);
}

async function postJson(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return parseResponse(response);
}

async function parseResponse(response) {
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || response.statusText);
  }
  return payload;
}

function setResultTitle(value) {
  state.lastTitle = value;
  $("resultTitle").textContent = value;
  renderAnswerWarning(null);
}

function renderAnswerWarning(message) {
  const node = $("answerWarning");
  if (!node) {
    return;
  }
  if (message) {
    node.textContent = `⚠️ ${message}`;
    node.hidden = false;
  } else {
    node.textContent = "";
    node.hidden = true;
  }
}

function renderInlineMarkdown(text) {
  // `text` is already HTML-escaped. Order matters: code first, then bold, then italic.
  return text
    .replace(/`([^`]+)`/g, (match, code) => `<code>${code}</code>`)
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*\n]+)\*/g, "<em>$1</em>")
    .replace(
      /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>',
    );
}

// Minimal, safe Markdown -> HTML for the answer panel. All source text is
// HTML-escaped before any tags are added, so model output cannot inject markup.
function renderMarkdown(source) {
  const lines = String(source).replace(/\r\n/g, "\n").split("\n");
  const html = [];
  let paragraph = [];

  const flushParagraph = () => {
    if (paragraph.length) {
      html.push(`<p>${paragraph.join("<br>")}</p>`);
      paragraph = [];
    }
  };

  let i = 0;
  while (i < lines.length) {
    const line = lines[i];

    if (/^\s*```/.test(line)) {
      flushParagraph();
      i += 1;
      const code = [];
      while (i < lines.length && !/^\s*```/.test(lines[i])) {
        code.push(lines[i]);
        i += 1;
      }
      i += 1; // skip the closing fence
      html.push(`<pre class="code-block"><code>${escapeHtml(code.join("\n"))}</code></pre>`);
      continue;
    }

    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      flushParagraph();
      const level = Math.min(heading[1].length + 1, 6);
      html.push(`<h${level}>${renderInlineMarkdown(escapeHtml(heading[2].trim()))}</h${level}>`);
      i += 1;
      continue;
    }

    if (/^\s*([-*_])\1{2,}\s*$/.test(line)) {
      flushParagraph();
      html.push("<hr>");
      i += 1;
      continue;
    }

    if (/^\s*[-*]\s+/.test(line)) {
      flushParagraph();
      const items = [];
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) {
        items.push(
          `<li>${renderInlineMarkdown(escapeHtml(lines[i].replace(/^\s*[-*]\s+/, "")))}</li>`,
        );
        i += 1;
      }
      html.push(`<ul>${items.join("")}</ul>`);
      continue;
    }

    if (/^\s*\d+\.\s+/.test(line)) {
      flushParagraph();
      const startMatch = line.match(/^\s*(\d+)\./);
      const start = startMatch ? Number(startMatch[1]) : 1;
      const items = [];
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) {
        items.push(
          `<li>${renderInlineMarkdown(escapeHtml(lines[i].replace(/^\s*\d+\.\s+/, "")))}</li>`,
        );
        i += 1;
      }
      const startAttr = start > 1 ? ` start="${start}"` : "";
      html.push(`<ol${startAttr}>${items.join("")}</ol>`);
      continue;
    }

    if (/^\s*$/.test(line)) {
      flushParagraph();
      i += 1;
      continue;
    }

    paragraph.push(renderInlineMarkdown(escapeHtml(line)));
    i += 1;
  }
  flushParagraph();
  return html.join("\n");
}

function renderAnswer(text) {
  const node = $("answerText");
  if (!node) {
    return;
  }
  const value = text == null ? "" : String(text);
  node.innerHTML = value.trim() ? renderMarkdown(value) : "";
  renderMath(node);
}

// Render LaTeX math with KaTeX auto-render when it is loaded. Math is left as
// literal text (e.g. "$2^{8.7}$") if the CDN is unreachable, so it degrades safely.
function renderMath(node) {
  if (typeof window.renderMathInElement !== "function") {
    return;
  }
  try {
    window.renderMathInElement(node, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "\\[", right: "\\]", display: true },
        { left: "$", right: "$", display: false },
        { left: "\\(", right: "\\)", display: false },
      ],
      ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
      throwOnError: false,
    });
  } catch (error) {
    /* Leave the literal math text in place on failure. */
  }
}

function setLatency(value) {
  $("latencyValue").textContent = value == null ? "- ms" : `${formatScore(value)} ms`;
}

function renderRaw(payload) {
  state.lastResult = payload;
  $("rawJson").textContent = JSON.stringify(payload, null, 2);
}

function exportReport() {
  if (!state.lastResult) {
    renderError(new Error("Run a workflow before exporting a report."));
    return;
  }
  const blob = new Blob([buildReportMarkdown(state.lastResult)], { type: "text/markdown" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `adip-report-${new Date().toISOString().replaceAll(":", "-").replaceAll(".", "-")}.md`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function buildReportMarkdown(payload) {
  const quality = extractQuality(payload);
  const citations = extractCitations(payload);
  const trace = extractTrace(payload);
  const lines = [
    `# ${state.lastTitle}`,
    "",
    `Generated: ${new Date().toISOString()}`,
    `Latency: ${$("latencyValue").textContent}`,
    "",
    "## Answer",
    "",
    $("answerText").textContent || "-",
    "",
    "## Quality",
    "",
    ...qualityLines(quality),
    "",
    "## Citations",
    "",
    ...citationLines(citations),
    "",
    "## Trace",
    "",
    ...traceLines(trace),
    "",
    "## Raw JSON",
    "",
    "```json",
    JSON.stringify(payload, null, 2),
    "```",
    "",
  ];
  return lines.join("\n");
}

function extractQuality(payload) {
  if (payload.quality) return payload.quality;
  if (payload.trace && payload.trace.llmops) return payload.trace.llmops.quality;
  if (payload.state && payload.state.llmops) return payload.state.llmops.quality;
  return null;
}

function extractCitations(payload) {
  if (payload.retrieved) return payload.retrieved;
  if (payload.state && payload.state.retrieved) return payload.state.retrieved;
  if (payload.trace && payload.trace.retrieved) return payload.trace.retrieved;
  return [];
}

function extractTrace(payload) {
  if (payload.state && payload.state.trace) return payload.state.trace;
  if (payload.trace && payload.trace.trace) return payload.trace.trace;
  return [];
}

function qualityLines(quality) {
  if (!quality) return ["- No quality metrics recorded."];
  return [
    `- Fidelity score: ${formatScore(quality.fidelity_score)}`,
    `- Citation coverage: ${formatPercent(quality.citation_coverage)}`,
    `- Visible citations: ${safeDisplay(quality.visible_citation_count)}`,
    `- Unsupported claims: ${safeDisplay(quality.unsupported_sentence_count)}`,
    `- Answer claims: ${safeDisplay(quality.answer_sentence_count)}`,
    `- Evidence count: ${safeDisplay(quality.evidence_count)}`,
  ];
}

function citationLines(citations) {
  if (!citations.length) return ["- No citations recorded."];
  return citations.map((item, index) => {
    const chunk = item.chunk || {};
    const label = item.citation || chunk.filename || `Citation ${index + 1}`;
    return `- ${label} | score ${formatScore(item.score)} | ${snippet(chunk.text || "")}`;
  });
}

function traceLines(trace) {
  if (!trace.length) return ["- No trace events recorded."];
  return trace.map((event) => {
    const duration = event.duration_ms == null ? "-" : `${formatScore(event.duration_ms)} ms`;
    return `- ${event.node_name || "node"} | ${event.status || "status"} | ${duration}`;
  });
}

function formatScore(value) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) {
    return "-";
  }
  return Number(value).toFixed(3);
}

function formatPercent(value) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) {
    return "-";
  }
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function labelMetric(value) {
  return value ? String(value).replaceAll("_", " ") : "-";
}

function slugify(value) {
  const slug = value.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
  return slug || "model";
}

function snippet(text) {
  const compact = text.replace(/\s+/g, " ").trim();
  return compact.length > 190 ? `${compact.slice(0, 187)}...` : compact;
}

function safeCount(payload, key) {
  return payload && payload[key] !== undefined ? payload[key] : "-";
}

function safeDisplay(value) {
  return value === undefined || value === null ? "-" : String(value);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
