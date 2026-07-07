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
  watchColdStart();
  refreshBenchmark();
  refreshGenerationEval();
  refreshModelProfiles();
  refreshIndexedDocuments();
  refreshRawDocuments();
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
      if (state.activeMode === "eval") {
        loadEvaluation();
      }
    });
  });
}

function bindForms() {
  $("ragForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    if ($("ragCompareBackends").checked) {
      await runBackendComparison(event.currentTarget);
      return;
    }
    await runRequest(event.currentTarget, "/rag/query", buildRagPayload(), renderRagResult);
  });
  $("ragBackend").addEventListener("change", () => {
    $("ragIndexPath").value = indexPathForBackend($("ragBackend").value);
    refreshIndexedDocuments();
  });
  $("refreshEval").addEventListener("click", () => loadEvaluation());

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
  $("indexPath").addEventListener("change", () => {
    refreshIndexedDocuments();
    refreshRawDocuments();
  });
  $("refreshRawDocs").addEventListener("click", () => refreshRawDocuments());
  $("inputPath").addEventListener("change", () => refreshRawDocuments());
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

async function refreshRawDocuments() {
  const rawDir = $("inputPath").value.trim() || "data/raw";
  const indexPath = $("indexPath").value.trim() || "data/processed/vector_index";
  try {
    const payload = await getJson(
      `/documents?raw_dir=${encodeURIComponent(rawDir)}&index_path=${encodeURIComponent(indexPath)}`,
    );
    renderRawDocuments(payload);
  } catch (error) {
    console.warn(error);
  }
}

function renderRawDocuments(payload) {
  const list = $("rawDocList");
  list.innerHTML = "";
  const items = payload.items || [];
  if (!items.length) {
    const empty = document.createElement("li");
    empty.className = "doc-empty";
    empty.textContent = "No documents in the raw folder yet. Upload one above.";
    list.appendChild(empty);
  }
  items.forEach((doc) => {
    const item = document.createElement("li");
    item.className = "doc-item";
    const meta = document.createElement("span");
    meta.className = "doc-meta";
    meta.textContent = `${doc.filename} · ${formatBytes(doc.size_bytes)}`;
    const badge = document.createElement("span");
    badge.className = doc.indexed ? "doc-badge indexed" : "doc-badge pending";
    badge.textContent = doc.indexed ? "indexed" : "not indexed";
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "danger-action compact-action";
    remove.textContent = "Delete";
    remove.addEventListener("click", () => deleteRawDocument(doc.filename));
    item.append(meta, badge, remove);
    list.appendChild(item);
  });
  $("docStaleHint").classList.toggle("hidden", !payload.index_stale);
}

async function deleteRawDocument(filename) {
  const confirmed = window.confirm(
    `Delete ${filename}? The index keeps serving its chunks until you Rebuild Index.`,
  );
  if (!confirmed) {
    return;
  }
  const rawDir = $("inputPath").value.trim() || "data/raw";
  try {
    const response = await fetch(
      `/documents/${encodeURIComponent(filename)}?raw_dir=${encodeURIComponent(rawDir)}`,
      { method: "DELETE" },
    );
    const payload = await parseResponse(response);
    setResultTitle("Document Delete");
    setLatency(null);
    renderAnswer([`Status: ${payload.status}`, `File: ${payload.filename}`, payload.note].join("\n"));
    renderQuality(null);
    renderCitations([]);
    renderTrace([]);
    renderRaw(payload);
  } catch (error) {
    renderError(error);
  } finally {
    refreshRawDocuments();
  }
}

function formatBytes(size) {
  const bytes = Number(size || 0);
  if (bytes >= 1024 * 1024) {
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }
  if (bytes >= 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${bytes} B`;
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
    all_backends: true,
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
  const metrics = statePayload.metrics || {};
  setResultTitle("Agent Run");
  setLatency(payload.latency_ms);
  renderAnswer(agentAnswerText(statePayload));
  renderAnswerWarning(payload.answer_warning);
  renderQuality(payload.quality);
  renderCitations(statePayload.retrieved || []);
  renderTrace(statePayload.trace || [], {
    engine: metrics.workflow_engine,
    totalMs: metrics.workflow_duration_ms,
  });
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
  const lines = [
    `Status: ${payload.status}`,
    `Documents: ${ingestion.document_count}`,
    `Chunks: ${ingestion.chunk_count}`,
    `Backend: ${index.backend}`,
    `Index: ${index.index_path}`,
  ];
  (payload.additional_indexes || []).forEach((extra) => {
    lines.push(`Also rebuilt: ${extra.index_path} (${extra.backend})`);
  });
  renderAnswer(lines.join("\n"));
  renderQuality(null);
  renderCitations([]);
  renderTrace([]);
  renderRaw(payload);
  refreshBenchmark();
  refreshIndexedDocuments();
  refreshRawDocuments();
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
    refreshRawDocuments();
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
    renderTrace(trace.trace || [], {
      engine: trace.metrics ? trace.metrics.workflow_engine : null,
      totalMs: trace.metrics ? trace.metrics.workflow_duration_ms : null,
    });
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

function renderTrace(events, meta) {
  const list = $("traceList");
  list.innerHTML = "";
  if (!events.length) {
    list.appendChild(emptyState("No trace events"));
    return;
  }

  if (meta && meta.engine) {
    const badgeRow = document.createElement("div");
    badgeRow.className = "trace-meta-row";
    const engineClass = meta.engine === "langgraph" ? "engine-badge langgraph" : "engine-badge";
    badgeRow.innerHTML = `
      <span class="${engineClass}">${escapeHtml(meta.engine)} engine</span>
      ${meta.totalMs == null ? "" : `<span class="trace-total">${formatScore(meta.totalMs)} ms total</span>`}
    `;
    list.appendChild(badgeRow);
  }

  const maxDuration = Math.max(
    1,
    ...events.map((event) => (event.duration_ms == null ? 0 : event.duration_ms))
  );
  events.forEach((event, position) => {
    const node = document.createElement("article");
    const failed = event.status === "failed";
    node.className = failed ? "trace-step failed" : "trace-step";
    const width = Math.max(2, Math.round(((event.duration_ms || 0) / maxDuration) * 100));
    node.innerHTML = `
      <div class="trace-step-head">
        <span class="trace-step-index">${position + 1}</span>
        <span class="trace-step-name">${escapeHtml(event.node_name || "node")}</span>
        <span class="trace-step-status">${escapeHtml(event.status || "status")}</span>
        <span class="trace-step-duration">${event.duration_ms == null ? "-" : `${formatScore(event.duration_ms)} ms`}</span>
      </div>
      <div class="trace-step-bar"><span style="width:${width}%"></span></div>
      <div class="trace-step-details hidden">
        ${event.error ? `<p class="trace-step-error">${escapeHtml(event.error)}</p>` : ""}
        <p class="item-meta">Retrieved ${safeCount(event.output_summary, "retrieved_count")} | Citations ${safeCount(event.output_summary, "citation_count")} | Plan steps ${safeCount(event.output_summary, "plan_step_count")} | Answer chars ${safeCount(event.output_summary, "answer_char_count")}</p>
        <p class="item-meta">Status after node: ${escapeHtml(stateField(event.output_summary, "status"))} | Task: ${escapeHtml(stateField(event.output_summary, "task_type"))}</p>
      </div>
    `;
    node.addEventListener("click", () => {
      node.querySelector(".trace-step-details").classList.toggle("hidden");
    });
    list.appendChild(node);
  });
}

function stateField(summary, key) {
  if (!summary || summary[key] == null) {
    return "-";
  }
  return String(summary[key]);
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
  showResultView("standard");
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

// ---------------------------------------------------------------------------
// Result-panel view switching (standard answer view / backend comparison /
// evaluation overview share the same panel).

function showResultView(view) {
  $("standardResult").classList.toggle("hidden", view !== "standard");
  $("compareView").classList.toggle("hidden", view !== "compare");
  $("evalView").classList.toggle("hidden", view !== "eval");
}

// ---------------------------------------------------------------------------
// Cold-start handling: the free hosting tier spins the instance down when idle,
// so the first page load can take ~a minute. Poll /health with patience, keep a
// banner up while waking, and unlock the page when the service responds.

async function watchColdStart() {
  const banner = $("coldStartBanner");
  const started = Date.now();
  const maxWaitMs = 180000;
  let firstFailure = true;

  const attempt = async () => {
    try {
      await refreshHealth();
      const status = $("serviceStatus");
      if (status.classList.contains("ok")) {
        banner.classList.add("hidden");
        setActionsDisabled(false);
        return;
      }
      throw new Error("not ready");
    } catch (error) {
      if (firstFailure) {
        firstFailure = false;
        banner.classList.remove("hidden");
        setActionsDisabled(true);
      }
      if (Date.now() - started < maxWaitMs) {
        setTimeout(attempt, 5000);
      } else {
        banner.textContent = "The service is not responding. Refresh the page to retry.";
      }
    }
  };
  await attempt();
}

function setActionsDisabled(disabled) {
  document.querySelectorAll(".primary-action").forEach((button) => {
    button.disabled = disabled;
  });
}

// ---------------------------------------------------------------------------
// Retrieval backend comparison: fire the same question at the tfidf, dense, and
// hybrid indexes in parallel and render the ranked lists side by side.

const INDEX_PATH_BY_BACKEND = {
  tfidf: "data/processed/vector_index",
  dense: "data/processed/vector_index_dense",
  hybrid: "data/processed/vector_index_hybrid",
};
const BACKEND_LABELS = {
  tfidf: "TF-IDF (sparse)",
  dense: "Dense LSA (semantic)",
  hybrid: "Hybrid (BM25 + dense)",
};

function indexPathForBackend(backend) {
  return INDEX_PATH_BY_BACKEND[backend] || INDEX_PATH_BY_BACKEND.tfidf;
}

async function runBackendComparison(form) {
  const button = form.querySelector("button[type='submit']");
  const originalText = button.innerHTML;
  button.disabled = true;
  button.innerHTML = '<span class="button-icon">...</span> Comparing';
  setResultTitle("Comparing Backends");

  const base = buildRagPayload();
  const backends = ["tfidf", "dense", "hybrid"];
  const started = performance.now();
  try {
    const results = await Promise.all(
      backends.map(async (backend) => {
        try {
          const payload = await postJson("/rag/query", {
            ...base,
            index_path: indexPathForBackend(backend),
          });
          return { backend, payload };
        } catch (error) {
          return { backend, error };
        }
      })
    );
    renderComparison(base.question, results);
    setLatency(performance.now() - started);
  } finally {
    button.disabled = false;
    button.innerHTML = originalText;
    refreshHealth();
  }
}

function renderComparison(question, results) {
  setResultTitle("Backend Comparison");
  showResultView("compare");
  const view = $("compareView");
  const reference = results[0] && results[0].payload ? results[0].payload.retrieved || [] : [];
  const referenceIds = reference.map((item) => (item.chunk || {}).chunk_id);

  const columns = results
    .map(({ backend, payload, error }) => {
      const label = BACKEND_LABELS[backend] || backend;
      if (error) {
        return `
          <div class="compare-column">
            <h3>${escapeHtml(label)}</h3>
            <p class="compare-error">${escapeHtml(error.message || "Query failed")}. If this index is missing, rebuild it from the Documents tab (backend: ${escapeHtml(backend)}).</p>
          </div>`;
      }
      const items = payload.retrieved || [];
      const maxScore = Math.max(0.0001, ...items.map((item) => item.score || 0));
      const rows = items
        .map((item) => {
          const chunk = item.chunk || {};
          const differs =
            backend !== "tfidf" && referenceIds[item.rank - 1] !== undefined &&
            referenceIds[item.rank - 1] !== chunk.chunk_id;
          const width = Math.max(3, Math.round(((item.score || 0) / maxScore) * 100));
          return `
            <article class="compare-item${differs ? " differs" : ""}">
              <p class="item-title">#${item.rank} ${escapeHtml(chunk.filename || "source")} <span class="compare-page">p.${escapeHtml(String(chunk.page_number == null ? "?" : chunk.page_number))}</span>${differs ? '<span class="diff-flag">differs</span>' : ""}</p>
              <div class="compare-score-bar"><span style="width:${width}%"></span></div>
              <p class="item-meta">score ${formatScore(item.score)}</p>
              <p class="item-meta">${escapeHtml(snippet(chunk.text || ""))}</p>
            </article>`;
        })
        .join("");
      return `
        <div class="compare-column">
          <h3>${escapeHtml(label)}</h3>
          <p class="compare-meta">${payload.latency_ms == null ? "" : `${formatScore(payload.latency_ms)} ms`} | reranker ${escapeHtml(payload.reranker || "none")}</p>
          ${rows || '<p class="compare-error">No results above the score floor.</p>'}
        </div>`;
    })
    .join("");

  view.innerHTML = `
    <div class="compare-head">
      <h3>Same question, three retrieval strategies</h3>
      <p class="item-meta">${escapeHtml(question)}</p>
      <p class="compare-note">Rows flagged <span class="diff-flag">differs</span> rank a different chunk than TF-IDF at the same position. Scores are not comparable across backends (cosine vs normalized RRF), so each column's bars are scaled to its own top score.</p>
    </div>
    <div class="compare-grid">${columns}</div>
  `;
  renderRaw(results.map(({ backend, payload, error }) => ({ backend, error: error ? String(error.message || error) : null, payload })));
}

// ---------------------------------------------------------------------------
// Evaluation tab: deterministic CI metrics + retrieval benchmark + offline
// judge/RAGAS snapshot, with agreement panels.

const CI_FLOORS = {
  gen_eval_mean_faithfulness: { label: "Faithfulness", bound: ">= 0.45" },
  gen_eval_grounded_rate: { label: "Grounded rate", bound: ">= 0.80" },
  gen_eval_mean_expected_coverage: { label: "Expected coverage", bound: ">= 0.65" },
  gen_eval_mean_answer_relevance: { label: "Answer relevance", bound: ">= 0.80" },
  gen_eval_mean_citation_coverage: { label: "Citation coverage", bound: ">= 0.55" },
  gen_eval_refusal_precision: { label: "Refusal precision", bound: ">= 0.80" },
  gen_eval_refusal_recall: { label: "Refusal recall", bound: ">= 0.40" },
  gen_eval_refusal_rate: { label: "Refusal rate", bound: "<= 0.20" },
};

async function loadEvaluation() {
  setResultTitle("Evaluation Overview");
  showResultView("eval");
  const view = $("evalView");
  view.innerHTML = '<p class="item-meta">Loading evaluation data…</p>';

  const [generation, benchmark, offline] = await Promise.all([
    getJson("/monitoring/generation-eval").catch(() => ({ available: false })),
    getJson("/monitoring/retrieval-benchmark").catch(() => ({ available: false })),
    getJson("/monitoring/offline-eval").catch(() => ({ available: false })),
  ]);

  view.innerHTML = [
    renderDeterministicPanel(generation),
    renderBenchmarkPanel(benchmark),
    renderJudgePanel(offline),
    renderRagasPanel(offline),
    renderAgreementPanel(generation, offline),
  ].join("");
  renderRaw({ generation, benchmark, offline });
}

function evalTile(label, value, note) {
  return `
    <div class="eval-tile">
      <span>${escapeHtml(label)}</span>
      <strong>${value}</strong>
      ${note ? `<em>${escapeHtml(note)}</em>` : ""}
    </div>`;
}

function renderDeterministicPanel(generation) {
  if (!generation.available) {
    return '<section class="eval-panel"><h3>Deterministic metrics (CI-gated)</h3><p class="item-meta">Not available — run the generation eval to populate.</p></section>';
  }
  const metrics = generation.metrics || {};
  const tiles = Object.entries(CI_FLOORS)
    .map(([key, spec]) =>
      evalTile(spec.label, formatScore(metrics[key]), `CI ${spec.bound}`)
    )
    .join("");
  return `
    <section class="eval-panel">
      <h3>Deterministic metrics (CI-gated)</h3>
      <p class="eval-panel-note">Recomputed on every build with the extractive writer over the real-document corpus (${safeDisplay(metrics.gen_eval_case_count)} cases). A pull request that pushes any value past its bound cannot merge.</p>
      <div class="eval-grid">${tiles}</div>
    </section>`;
}

function renderBenchmarkPanel(benchmark) {
  if (!benchmark.available) {
    return '<section class="eval-panel"><h3>Retrieval benchmark</h3><p class="item-meta">Not available — run the retrieval benchmark to populate.</p></section>';
  }
  const metrics = benchmark.metrics || {};
  const variants = ["tfidf", "dense_lsa", "hybrid"]
    .map((variant) => {
      const mrr = metrics[`${variant}_mrr`];
      if (mrr === undefined) {
        return "";
      }
      return evalTile(labelMetric(variant), formatScore(mrr), "MRR");
    })
    .join("");
  return `
    <section class="eval-panel">
      <h3>Retrieval benchmark</h3>
      <p class="eval-panel-note">Best plain backend: <strong>${escapeHtml(labelMetric(benchmark.best_backend_by_mrr))}</strong> · best variant: <strong>${escapeHtml(labelMetric(benchmark.best_variant_by_mrr))}</strong>. Hit rate is saturated at 1.0 on this corpus, so MRR (and the graded RAGAS context metrics below) carry the signal.</p>
      <div class="eval-grid">${variants || evalTile("Variants", safeDisplay(metrics.variant_count), "in last benchmark")}</div>
    </section>`;
}

function renderJudgePanel(offline) {
  if (!offline.available || !offline.judge) {
    return '<section class="eval-panel"><h3>LLM-as-judge (offline)</h3><p class="item-meta">No judge snapshot committed. Run run_generation_eval with --judge-model-name and commit the snapshot.</p></section>';
  }
  const judge = offline.judge;
  const inter = offline.inter_judge || {};
  return `
    <section class="eval-panel">
      <h3>LLM-as-judge (offline · ${escapeHtml(judge.judge_model || "judge")} · ${escapeHtml(judge.run_date || "")})</h3>
      <div class="eval-grid">
        ${evalTile("Judge faithfulness", formatScore(judge.judge_mean_faithfulness), `${safeDisplay(judge.judged_count)}/${safeDisplay(judge.total_answered)} answers judged`)}
        ${evalTile("Judge relevance", formatScore(judge.judge_mean_relevance), "the writer's real weakness")}
        ${evalTile("Gap vs lexical proxy", formatScore(judge.judge_lexical_faithfulness_gap), "proxy underestimates")}
        ${evalTile("Inter-judge relevance r", formatScore(inter.relevance_correlation), `vs ${(inter.second_judge_model || "").split(" ")[0]}`)}
      </div>
      <p class="eval-panel-note">${escapeHtml(judge.finding || "")}</p>
    </section>`;
}

function renderRagasPanel(offline) {
  if (!offline.available || !offline.ragas) {
    return '<section class="eval-panel"><h3>RAGAS (offline)</h3><p class="item-meta">No RAGAS snapshot committed. Install the [ragas] extra and run run_generation_eval with --ragas-model-name.</p></section>';
  }
  const ragas = offline.ragas;
  return `
    <section class="eval-panel">
      <h3>RAGAS (offline · ${escapeHtml(ragas.ragas_llm || "LLM")} · ${escapeHtml(ragas.run_date || "")})</h3>
      <div class="eval-grid">
        ${evalTile("Faithfulness", formatScore(ragas.ragas_mean_faithfulness), "claim decomposition")}
        ${evalTile("Answer relevancy", formatScore(ragas.ragas_mean_answer_relevancy), "see caveat")}
        ${evalTile("Context precision", formatScore(ragas.ragas_mean_context_precision), "graded retrieval")}
        ${evalTile("Context recall", formatScore(ragas.ragas_mean_context_recall), "graded retrieval")}
      </div>
      <p class="eval-panel-note"><strong>Finding:</strong> ${escapeHtml(ragas.finding || "")}</p>
      <p class="eval-panel-note"><strong>Caveat:</strong> ${escapeHtml(ragas.answer_relevancy_caveat || "")}</p>
    </section>`;
}

function renderAgreementPanel(generation, offline) {
  const lexical = generation.available ? generation.faithfulness : null;
  const judge = offline.available && offline.judge ? offline.judge.judge_mean_faithfulness : null;
  const ragas = offline.available && offline.ragas ? offline.ragas.ragas_mean_faithfulness : null;
  if (lexical == null && judge == null && ragas == null) {
    return "";
  }
  return `
    <section class="eval-panel">
      <h3>Three scorers, one story: faithfulness</h3>
      <div class="eval-grid">
        ${evalTile("Lexical proxy (CI)", formatScore(lexical), "token overlap — deterministic")}
        ${evalTile("LLM judge", formatScore(judge), "semantic — frontier model")}
        ${evalTile("RAGAS", formatScore(ragas), "claim decomposition — local 8B")}
      </div>
      <p class="eval-panel-note">The judge and RAGAS agree the extractive writer fabricates nothing (~0.93–0.97); the lexical proxy reads low (~0.59) because verbatim answers carry formatting tokens. Measuring that disagreement — instead of trusting any single number — is the point of this tab.</p>
    </section>`;
}
