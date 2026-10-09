/**
 * BiggBoss CCTV Multi-Stream Video Intelligence
 * Client Application Logic (Vanilla JavaScript)
 */

document.addEventListener("DOMContentLoaded", () => {
  initNavigation();
  initSystemStatus();
  initChatbot();
  initReID();
  initSearch();
  initAlerts();
  initPrivacy();
  initResearch();
  initIngestion();
  initModals();
});

/* ==========================================================================
   GLOBAL STATE & UTILS
   ========================================================================== */
const AppState = {
  currentView: "chatbot",
  status: null,
  ingestResults: null,
  activeVideos: [],
  privacyEnabled: false,
};

function showToast(message, type = "info") {
  const container = document.getElementById("toast-container");
  if (!container) return;
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `<span>${message}</span>`;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = "0";
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}

function openImageModal(imgSrc, title = "Visual Evidence Crop") {
  const modal = document.getElementById("image-modal");
  const modalImg = document.getElementById("modal-img");
  const modalTitle = document.getElementById("modal-img-title");
  if (modal && modalImg) {
    modalImg.src = imgSrc;
    if (modalTitle) modalTitle.textContent = title;
    modal.classList.add("active");
  }
}

/* ==========================================================================
   NAVIGATION
   ========================================================================== */
function initNavigation() {
  const navItems = document.querySelectorAll(".nav-item");
  const views = document.querySelectorAll(".view-section");
  const topTitle = document.getElementById("current-view-title");
  const topBadge = document.getElementById("current-view-badge");

  const viewTitles = {
    chatbot: { title: "CCTV Intelligence Chatbot", badge: "GPT4All & Grounded Agent" },
    reid: { title: "Cross-Camera Re-ID Explorer", badge: "SigLIP Multi-Camera" },
    search: { title: "Single-Stream Natural Search", badge: "Open-Vocab Retrieval" },
    alerts: { title: "Standing Queries & Live Alerts", badge: "Automated Triggers" },
    privacy: { title: "On-Premise Privacy Filter", badge: "Zero Cloud Redaction" },
    research: { title: "Research Benchmark & Ablation", badge: "20% Rubric Contribution" },
    ingest: { title: "Camera Streams & Ingestion", badge: "Multi-Source Feeds" },
  };

  navItems.forEach((item) => {
    item.addEventListener("click", () => {
      const targetView = item.dataset.view;
      if (!targetView) return;

      navItems.forEach((n) => n.classList.remove("active"));
      item.classList.add("active");

      views.forEach((v) => {
        if (v.id === `view-${targetView}`) {
          v.classList.add("active");
        } else {
          v.classList.remove("active");
        }
      });

      AppState.currentView = targetView;
      if (topTitle && viewTitles[targetView]) {
        topTitle.textContent = viewTitles[targetView].title;
      }
      if (topBadge && viewTitles[targetView]) {
        topBadge.textContent = viewTitles[targetView].badge;
      }

      // Refresh view-specific data
      if (targetView === "reid") loadReIDData();
      if (targetView === "alerts") loadAlertsData();
      if (targetView === "privacy") loadPrivacyData();
      if (targetView === "research") loadResearchData();
      if (targetView === "ingest") loadIngestData();
    });
  });
}

/* ==========================================================================
   SYSTEM STATUS & TOPBAR
   ========================================================================== */
async function initSystemStatus() {
  const privacyBtn = document.getElementById("privacy-toggle-btn");
  if (privacyBtn) {
    privacyBtn.addEventListener("click", async () => {
      try {
        const res = await fetch("/api/privacy/toggle", { method: "POST" });
        const data = await res.json();
        AppState.privacyEnabled = data.enabled;
        updatePrivacyUI(data.enabled);
        showToast(
          data.enabled ? "Privacy Redaction Activated" : "Privacy Redaction Disabled",
          data.enabled ? "success" : "warning"
        );
        // Refresh current view if needed
        if (AppState.currentView === "reid") loadReIDData();
        if (AppState.currentView === "privacy") loadPrivacyData();
      } catch (err) {
        showToast("Failed to toggle privacy", "danger");
      }
    });
  }

  await refreshSystemStatus();
  setInterval(refreshSystemStatus, 10000);
}

function updatePrivacyUI(enabled) {
  const privacyBtn = document.getElementById("privacy-toggle-btn");
  const privacyText = document.getElementById("privacy-toggle-text");
  if (privacyBtn) {
    if (enabled) {
      privacyBtn.classList.add("active");
      if (privacyText) privacyText.textContent = "Privacy Redaction ON";
    } else {
      privacyBtn.classList.remove("active");
      if (privacyText) privacyText.textContent = "Privacy Redaction OFF";
    }
  }
}

async function refreshSystemStatus() {
  try {
    const res = await fetch("/api/status");
    if (!res.ok) return;
    const data = await res.json();
    AppState.status = data;
    AppState.privacyEnabled = data.privacy_enabled;

    // Update Topbar Stats
    const statCams = document.getElementById("stat-cams");
    const statEntities = document.getElementById("stat-entities");
    const statHandovers = document.getElementById("stat-handovers");
    const sysDot = document.getElementById("system-status-dot");
    const sysText = document.getElementById("system-status-text");

    if (statCams) statCams.textContent = data.camera_count;
    if (statEntities) statEntities.textContent = data.entity_count;
    if (statHandovers) statHandovers.textContent = data.handover_count;

    if (sysText) {
      sysText.textContent = data.models_ready
        ? `AI Models Active (${data.device.toUpperCase()})`
        : "Models Standby";
    }
    if (sysDot) {
      sysDot.style.background = data.models_ready ? "var(--accent-success)" : "var(--accent-warning)";
    }

    updatePrivacyUI(data.privacy_enabled);
  } catch (err) {
    console.warn("Status check failed", err);
  }
}

/* ==========================================================================
   CHATBOT LOGIC
   ========================================================================== */
function initChatbot() {
  const chatInput = document.getElementById("chat-query-input");
  const sendBtn = document.getElementById("chat-send-btn");
  const chips = document.querySelectorAll(".query-chip");
  const clearBtn = document.getElementById("chat-clear-btn");
  const gpt4allSelect = document.getElementById("gpt4all-model-select");
  const loadGpt4allBtn = document.getElementById("load-gpt4all-btn");

  if (sendBtn && chatInput) {
    sendBtn.addEventListener("click", () => sendChatMessage());
    chatInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendChatMessage();
      }
    });
  }

  chips.forEach((chip) => {
    chip.addEventListener("click", () => {
      const query = chip.dataset.query || chip.textContent.trim();
      if (chatInput) chatInput.value = query;
      sendChatMessage();
    });
  });

  if (clearBtn) {
    clearBtn.addEventListener("click", async () => {
      await fetch("/api/chat/clear", { method: "POST" });
      const container = document.getElementById("chat-messages-container");
      if (container) {
        container.innerHTML = `
          <div class="chat-welcome">
            <div class="chat-welcome-icon">
              <svg width="28" height="28" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"/></svg>
            </div>
            <h3>BiggBoss CCTV Intelligence Chatbot</h3>
            <p>Ask natural questions about what happened across your camera feeds. All answers are authoritatively grounded in timestamps and visual crops.</p>
          </div>
        `;
      }
    });
  }

  // Engine switcher change
  const engineRadios = document.querySelectorAll('input[name="chat-engine"]');
  engineRadios.forEach((r) => {
    r.addEventListener("change", (e) => {
      const gpt4allConfig = document.getElementById("gpt4all-config-panel");
      if (gpt4allConfig) {
        gpt4allConfig.style.display = e.target.value === "gpt4all" ? "block" : "none";
      }
    });
  });

  if (loadGpt4allBtn && gpt4allSelect) {
    loadGpt4allBtn.addEventListener("click", async () => {
      const model = gpt4allSelect.value;
      loadGpt4allBtn.disabled = true;
      loadGpt4allBtn.innerHTML = `<span>Loading...</span>`;
      try {
        const res = await fetch("/api/gpt4all/load", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model_name: model }),
        });
        const data = await res.json();
        showToast(`Loaded ${data.loaded_model} on CPU!`, "success");
        const statusSpan = document.getElementById("gpt4all-status-badge");
        if (statusSpan) statusSpan.textContent = `Active: ${model}`;
      } catch (err) {
        showToast("Failed to load GPT4All model", "danger");
      } finally {
        loadGpt4allBtn.disabled = false;
        loadGpt4allBtn.innerHTML = `<span>Load Model</span>`;
      }
    });
  }

  // Quick Index Trigger inside Chatbot view if not indexed
  const quickIndexBtn = document.getElementById("quick-index-chat-btn");
  if (quickIndexBtn) {
    quickIndexBtn.addEventListener("click", async () => {
      quickIndexBtn.disabled = true;
      quickIndexBtn.innerHTML = `<span>Indexing Feeds...</span>`;
      try {
        const res = await fetch("/api/ingest/quick", { method: "POST" });
        const data = await res.json();
        showToast(`Indexed ${data.camera_count} cameras and ${data.vehicles_count} vehicles!`, "success");
        await refreshSystemStatus();
        document.getElementById("chat-indexing-notice").style.display = "none";
      } catch (e) {
        showToast("Quick index failed", "danger");
      } finally {
        quickIndexBtn.disabled = false;
        quickIndexBtn.innerHTML = `<span>Index Project Feeds</span>`;
      }
    });
  }

  // Knowledge base manager
  initKBManager();
}

async function sendChatMessage() {
  const input = document.getElementById("chat-query-input");
  const query = input.value.trim();
  if (!query) return;

  const container = document.getElementById("chat-messages-container");
  const sendBtn = document.getElementById("chat-send-btn");

  // Remove welcome placeholder if present
  const welcome = container.querySelector(".chat-welcome");
  if (welcome) welcome.remove();

  // Append user bubble
  const userBubble = document.createElement("div");
  userBubble.className = "chat-bubble-user";
  userBubble.textContent = query;
  container.appendChild(userBubble);
  input.value = "";
  container.scrollTop = container.scrollHeight;

  // Append loading assistant bubble
  const assistantBubble = document.createElement("div");
  assistantBubble.className = "chat-bubble-assistant";
  assistantBubble.innerHTML = `
    <div class="assistant-meta">
      <span class="badge-pill badge-indigo">Analyzing Footage...</span>
    </div>
    <div class="assistant-text">Querying open-vocabulary detector and SigLIP spatio-temporal embeddings...</div>
  `;
  container.appendChild(assistantBubble);
  container.scrollTop = container.scrollHeight;

  sendBtn.disabled = true;

  try {
    const useGpt4all = document.querySelector('input[name="chat-engine"]:checked')?.value === "gpt4all";
    const gpt4allModel = document.getElementById("gpt4all-model-select")?.value || "Llama-3.2-1B-Instruct-Q4_0.gguf";

    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: query,
        use_gpt4all: useGpt4all,
        gpt4all_model: gpt4allModel,
      }),
    });

    if (!res.ok) {
      const err = await res.json();
      assistantBubble.innerHTML = `
        <div class="assistant-meta">
          <span class="badge-pill badge-amber">Notice</span>
        </div>
        <div class="assistant-text">${err.detail || "Error querying camera intelligence."}</div>
      `;
      return;
    }

    const data = await res.json();

    // Check if clarification is required
    if (data.needs_clarification) {
      renderClarificationCard(assistantBubble, data, query);
    } else {
      renderAssistantResponse(assistantBubble, data);
    }
  } catch (err) {
    assistantBubble.innerHTML = `
      <div class="assistant-meta"><span class="badge-pill badge-amber">Error</span></div>
      <div class="assistant-text">Failed to connect to backend engine: ${err.message}</div>
    `;
  } finally {
    sendBtn.disabled = false;
    container.scrollTop = container.scrollHeight;
  }
}

function renderAssistantResponse(bubble, data) {
  const modelBadge = `<span class="badge-pill badge-indigo">${data.model_used || "Local Grounded Agent"}</span>`;

  let primaryHtml = "";
  if (data.primary_match) {
    const pm = data.primary_match;
    let eventBadge = `<span class="badge-pill badge-cyan">Sighting</span>`;
    if (pm.event_type === "left") eventBadge = `<span class="badge-pill badge-amber">Confirmed Departure</span>`;
    if (pm.event_type === "entered") eventBadge = `<span class="badge-pill badge-emerald">Confirmed Entry</span>`;

    primaryHtml = `
      <div class="evidence-card">
        <div class="evidence-crop-box" onclick="openImageModal('${pm.crop}', '${pm.camera_name} @ ${pm.timestamp_str}')">
          <img src="${pm.crop}" alt="Visual Evidence" />
        </div>
        <div class="evidence-details">
          <div class="evidence-header">
            <strong style="color: #fff; font-size: 0.92rem;">${pm.camera_name}</strong>
            ${eventBadge}
          </div>
          <div class="evidence-facts-grid">
            <div class="fact-item">Timestamp: <strong>${pm.timestamp.toFixed(2)}s (${pm.timestamp_str})</strong></div>
            <div class="fact-item">Visual Match: <strong>${(pm.confidence * 100).toFixed(1)}%</strong></div>
            <div class="fact-item">Subject: <strong>${pm.dominant_color} ${pm.label}</strong></div>
            <div class="fact-item">Classification: <strong>${pm.event_type.toUpperCase()}</strong></div>
          </div>
          <p style="font-size: 0.78rem; color: var(--text-muted); margin-top: 0.3rem;">${pm.details || ""}</p>
        </div>
      </div>
    `;
  }

  let timelineHtml = "";
  if (data.timeline_steps && data.timeline_steps.length > 0) {
    let stepsCards = "";
    data.timeline_steps.forEach((step, idx) => {
      stepsCards += `
        <div class="timeline-step-card">
          <img class="timeline-step-crop" src="${step.crop}" onclick="openImageModal('${step.crop}', '${step.camera}')" />
          <div class="timeline-step-cam">Step ${idx + 1}: ${step.camera}</div>
          <div class="timeline-step-time">${step.start_time.toFixed(1)}s – ${step.end_time.toFixed(1)}s</div>
          <div style="font-size: 0.68rem; color: var(--text-secondary);">${step.status === "exit" ? "Exited" : "Sighted"}</div>
        </div>
      `;
    });

    timelineHtml = `
      <div class="timeline-strip">
        <div class="timeline-strip-title">Cross-Camera Timeline Continuity</div>
        <div class="timeline-steps-grid">${stepsCards}</div>
      </div>
    `;
  }

  bubble.innerHTML = `
    <div class="assistant-meta">${modelBadge}</div>
    <div class="assistant-text">${data.content}</div>
    ${primaryHtml}
    ${timelineHtml}
  `;
}

function renderClarificationCard(bubble, data, originalQuery) {
  let camOptions = "";
  (data.available_cameras || ["Gate Cam", "Rear Cam"]).forEach((cam) => {
    camOptions += `<option value="${cam}">${cam}</option>`;
  });

  bubble.innerHTML = `
    <div class="assistant-meta"><span class="badge-pill badge-amber">Clarify-Once Spatial Memory</span></div>
    <div class="assistant-text">${data.content}</div>
    <div class="clarify-alert-box">
      <h4>
        <svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>
        Spatial Ambiguity Detected: "${data.clarification_entity}"
      </h4>
      <p style="font-size: 0.85rem; color: #fef3c7; margin-bottom: 0.8rem;">${data.clarification_prompt}</p>
      <div style="display: flex; gap: 0.8rem; align-items: center;">
        <select id="clarify-cam-select" class="select-field" style="max-width: 200px;">${camOptions}</select>
        <button id="clarify-save-btn" class="btn btn-primary btn-sm">Save to Memory & Answer</button>
      </div>
    </div>
  `;

  const saveBtn = bubble.querySelector("#clarify-save-btn");
  if (saveBtn) {
    saveBtn.addEventListener("click", async () => {
      const selectedCam = bubble.querySelector("#clarify-cam-select").value;
      saveBtn.disabled = true;
      saveBtn.textContent = "Saving...";

      await fetch("/api/kb/learn", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          referent: data.clarification_entity,
          target: selectedCam,
        }),
      });

      showToast(`Learned: "${data.clarification_entity}" maps to ${selectedCam}`, "success");
      loadKBTable();

      // Automatically re-execute the original query
      const chatInput = document.getElementById("chat-query-input");
      if (chatInput) chatInput.value = originalQuery;
      sendChatMessage();
    });
  }
}

async function initKBManager() {
  const addBtn = document.getElementById("kb-add-btn");
  const refInput = document.getElementById("kb-new-referent");
  const tgtInput = document.getElementById("kb-new-target");

  if (addBtn && refInput && tgtInput) {
    addBtn.addEventListener("click", async () => {
      const ref = refInput.value.trim();
      const tgt = tgtInput.value.trim();
      if (!ref || !tgt) {
        showToast("Please provide both referent and target camera", "warning");
        return;
      }
      await fetch("/api/kb/learn", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ referent: ref, target: tgt }),
      });
      refInput.value = "";
      tgtInput.value = "";
      showToast(`Added mapping: "${ref}" ➔ ${tgt}`, "success");
      loadKBTable();
    });
  }

  loadKBTable();
}

async function loadKBTable() {
  const container = document.getElementById("kb-mappings-list");
  if (!container) return;

  try {
    const res = await fetch("/api/kb");
    const data = await res.json();
    const mappings = data.mappings || {};

    if (Object.keys(mappings).length === 0) {
      container.innerHTML = `<p style="font-size: 0.8rem; color: var(--text-muted);">No custom spatial referents learned yet.</p>`;
      return;
    }

    let rows = "";
    for (const [ref, tgt] of Object.entries(mappings)) {
      rows += `
        <div style="display: flex; align-items: center; justify-content: space-between; padding: 0.4rem 0.6rem; background: rgba(15,23,42,0.6); border-radius: 6px; font-size: 0.8rem;">
          <div><code style="color: var(--accent-primary);">'${ref}'</code> ➔ <strong style="color: #fff;">${tgt}</strong></div>
          <button class="btn btn-outline btn-sm" style="padding: 0.2rem 0.5rem;" onclick="deleteKBMapping('${ref}')">Remove</button>
        </div>
      `;
    }
    container.innerHTML = rows;
  } catch (err) {
    console.warn("KB load failed", err);
  }
}

window.deleteKBMapping = async function (ref) {
  await fetch(`/api/kb/${encodeURIComponent(ref)}`, { method: "DELETE" });
  showToast(`Deleted referent '${ref}'`, "info");
  loadKBTable();
};

/* ==========================================================================
   CROSS-CAMERA RE-ID & JOURNEY EXPLORER
   ========================================================================== */
function initReID() {
  const filterBtns = document.querySelectorAll(".filter-pill");
  const searchInput = document.getElementById("reid-search-input");

  filterBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      filterBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      renderFilteredHandovers();
    });
  });

  if (searchInput) {
    searchInput.addEventListener("input", () => renderFilteredHandovers());
  }
}

async function loadReIDData() {
  const feed = document.getElementById("reid-handovers-feed");
  if (!feed) return;

  try {
    const res = await fetch("/api/ingest/results");
    const data = await res.json();
    AppState.ingestResults = data;

    if (!data.indexed) {
      feed.innerHTML = `
        <div style="text-align: center; padding: 4rem 2rem; color: var(--text-secondary);">
          <h3>No Camera Feeds Indexed Yet</h3>
          <p style="margin: 0.6rem 0 1.2rem;">Index the project camera footage to explore vehicle handovers across feeds.</p>
          <button class="btn btn-primary" onclick="triggerQuickIndex()">Index Gate Cam & Rear Cam</button>
        </div>
      `;
      return;
    }

    // Update KPI Metrics
    document.getElementById("reid-metric-cams").textContent = data.cameras.length;
    document.getElementById("reid-metric-handovers").textContent = data.handovers.length;
    document.getElementById("reid-metric-vehicles").textContent = data.total_vehicles_count;
    document.getElementById("reid-metric-delay").textContent = `${data.average_handover_delay}s`;

    renderFilteredHandovers();
    renderJourneysList(data.journeys);
  } catch (err) {
    console.error("Re-ID data load failed", err);
  }
}

function renderFilteredHandovers() {
  const feed = document.getElementById("reid-handovers-feed");
  if (!feed || !AppState.ingestResults) return;

  const activeFilter = document.querySelector(".filter-pill.active")?.dataset.filter || "all";
  const searchVal = document.getElementById("reid-search-input")?.value.toLowerCase().trim() || "";

  let handovers = AppState.ingestResults.handovers || [];

  if (activeFilter === "sequential") {
    handovers = handovers.filter((h) => h.delay_seconds >= 0);
  } else if (activeFilter === "concurrent") {
    handovers = handovers.filter((h) => h.delay_seconds < 0);
  }

  if (searchVal) {
    handovers = handovers.filter((h) =>
      h.label.toLowerCase().includes(searchVal) ||
      h.dominant_color.toLowerCase().includes(searchVal) ||
      h.description.toLowerCase().includes(searchVal)
    );
  }

  if (handovers.length === 0) {
    feed.innerHTML = `<p style="text-align: center; padding: 3rem; color: var(--text-muted);">No handovers match the selected filter criteria.</p>`;
    return;
  }

  let html = "";
  handovers.forEach((ho) => {
    const isSeq = ho.delay_seconds >= 0;
    const transitionText = isSeq ? `Transition Delay: ${ho.delay_seconds.toFixed(1)}s` : "Concurrent Cross-View";

    html += `
      <div class="handover-card">
        <div class="handover-header">
          <div class="handover-vehicle-title">
            <svg width="22" height="22" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 19H5V8h14v11zM16 4H8v4h8V4z"/></svg>
            <span>Vehicle #${ho.global_id} · <span style="color: var(--accent-primary); text-transform: capitalize;">${ho.dominant_color} ${ho.label}</span></span>
          </div>
          <span class="badge-pill badge-emerald">${(ho.similarity * 100).toFixed(1)}% SigLIP Visual Match</span>
        </div>

        <div class="handover-flow-banner">
          <div>Left <strong>${ho.from_camera}</strong> at <span style="font-family: var(--font-mono);">${ho.exit_time.toFixed(1)}s</span></div>
          <div style="font-weight: 600; color: var(--accent-primary);">➔ ${transitionText} ➔</div>
          <div>Appeared on <strong>${ho.to_camera}</strong> at <span style="font-family: var(--font-mono);">${ho.entry_time.toFixed(1)}s</span></div>
        </div>

        <div class="handover-grid">
          <div class="cam-observation-box">
            <img class="cam-obs-crop" src="${ho.from_crop}" onclick="openImageModal('${ho.from_crop}', '${ho.from_camera}')" />
            <div class="cam-obs-meta">
              <span class="cam-obs-name">${ho.from_camera} (Exit)</span>
              <span class="cam-obs-time">${ho.exit_time.toFixed(1)}s</span>
            </div>
          </div>

          <div class="reid-connection-flow">
            <div class="similarity-circle">
              <span class="sim-pct">${(ho.similarity * 100).toFixed(0)}%</span>
              <span class="sim-label">Match</span>
            </div>
            <span style="font-size: 0.75rem; color: var(--text-secondary);">${isSeq ? `Δt = ${ho.delay_seconds.toFixed(1)}s` : "Multi-Angle"}</span>
          </div>

          <div class="cam-observation-box">
            <img class="cam-obs-crop" src="${ho.to_crop}" onclick="openImageModal('${ho.to_crop}', '${ho.to_camera}')" />
            <div class="cam-obs-meta">
              <span class="cam-obs-name">${ho.to_camera} (Entry)</span>
              <span class="cam-obs-time">${ho.entry_time.toFixed(1)}s</span>
            </div>
          </div>
        </div>
      </div>
    `;
  });

  feed.innerHTML = html;
}

function renderJourneysList(journeys) {
  const container = document.getElementById("reid-journeys-accordion");
  if (!container || !journeys) return;

  let html = "";
  journeys.forEach((j) => {
    let sightingsHtml = "";
    j.sightings.forEach((s) => {
      sightingsHtml += `
        <div style="background: rgba(11,18,30,0.8); border: 1px solid var(--border-card); border-radius: 8px; padding: 0.6rem; text-align: center; min-width: 130px;">
          <img src="${s.crop}" style="width: 100%; height: 80px; object-fit: cover; border-radius: 4px; margin-bottom: 0.4rem;" onclick="openImageModal('${s.crop}', '${s.camera}')" />
          <div style="font-weight: 600; font-size: 0.78rem; color: #fff;">${s.camera}</div>
          <div style="font-family: var(--font-mono); font-size: 0.7rem; color: var(--accent-primary);">${s.start_time.toFixed(1)}s – ${s.end_time.toFixed(1)}s</div>
        </div>
      `;
    });

    html += `
      <div style="background: var(--bg-surface); border: 1px solid var(--border-card); border-radius: 12px; padding: 1.2rem; margin-bottom: 1rem;">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.8rem;">
          <div style="display: flex; align-items: center; gap: 0.6rem;">
            <strong style="color: #fff; font-size: 0.95rem;">${j.title}</strong>
            <span class="badge-pill badge-cyan">${j.dominant_color} ${j.label}</span>
          </div>
          <span style="font-size: 0.78rem; color: var(--text-secondary);">${j.sightings.length} Camera Sightings</span>
        </div>
        <div style="display: flex; gap: 0.8rem; overflow-x: auto; padding-bottom: 0.4rem;">${sightingsHtml}</div>
      </div>
    `;
  });

  container.innerHTML = html;
}

/* ==========================================================================
   SINGLE-STREAM NATURAL SEARCH
   ========================================================================== */
function initSearch() {
  const indexBtn = document.getElementById("search-index-btn");
  const queryBtn = document.getElementById("search-query-btn");
  const queryInput = document.getElementById("search-query-input");
  const sampleSlider = document.getElementById("search-sample-slider");
  const sampleVal = document.getElementById("search-sample-val");

  if (sampleSlider && sampleVal) {
    sampleSlider.addEventListener("input", () => {
      sampleVal.textContent = sampleSlider.value;
    });
  }

  if (indexBtn) {
    indexBtn.addEventListener("click", async () => {
      const select = document.getElementById("search-clip-select");
      const videoPath = select ? select.value : "";
      const maxSamples = parseInt(sampleSlider ? sampleSlider.value : "32", 10);
      if (!videoPath) {
        showToast("Please choose a video clip to index", "warning");
        return;
      }

      indexBtn.disabled = true;
      indexBtn.innerHTML = `<span>Sampling Frames...</span>`;
      try {
        const res = await fetch("/api/search/index", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ video_path: videoPath, max_samples: maxSamples }),
        });
        const data = await res.json();
        showToast(`Indexed ${data.sample_count} frames (${data.duration}s video)!`, "success");
        document.getElementById("search-active-notice").innerHTML = `
          Clip Ready: <strong>${videoPath.split(/[\\/]/).pop()}</strong> (${data.sample_count} samples cached)
        `;
        document.getElementById("search-query-bar").style.display = "flex";
      } catch (err) {
        showToast("Single-stream indexing failed", "danger");
      } finally {
        indexBtn.disabled = false;
        indexBtn.innerHTML = `<span>Index Clip</span>`;
      }
    });
  }

  if (queryBtn && queryInput) {
    queryBtn.addEventListener("click", () => executeSingleSearch());
    queryInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") executeSingleSearch();
    });
  }
}

async function executeSingleSearch() {
  const queryInput = document.getElementById("search-query-input");
  const query = queryInput.value.trim();
  if (!query) return;

  const resultsGrid = document.getElementById("search-results-grid");
  const queryBtn = document.getElementById("search-query-btn");

  queryBtn.disabled = true;
  resultsGrid.innerHTML = `<div style="grid-column: 1/-1; text-align: center; padding: 3rem; color: var(--text-secondary);">Ranking frames with SigLIP open-vocabulary text embeddings...</div>`;

  try {
    const res = await fetch("/api/search/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: query, top_k: 6 }),
    });
    const data = await res.json();

    if (!data.matches || data.matches.length === 0) {
      resultsGrid.innerHTML = `<div style="grid-column: 1/-1; text-align: center; padding: 3rem; color: var(--text-muted);">No matching frames found for "${query}".</div>`;
      return;
    }

    let cards = "";
    data.matches.forEach((m, idx) => {
      cards += `
        <div style="background: var(--bg-card); border: 1px solid var(--border-card); border-radius: 12px; overflow: hidden; box-shadow: var(--shadow-sm);">
          <img src="${m.image}" style="width: 100%; height: 180px; object-fit: cover; cursor: pointer;" onclick="openImageModal('${m.image}', 'Match @ ${m.timestamp_str}')" />
          <div style="padding: 1rem;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.4rem;">
              <span class="badge-pill badge-cyan">Rank #${idx + 1}</span>
              <span style="font-family: var(--font-mono); font-weight: 600; color: var(--accent-success);">${(m.score * 100).toFixed(1)}% Score</span>
            </div>
            <div style="font-size: 0.8rem; color: var(--text-secondary);">Timestamp: <strong style="color: #fff;">${m.timestamp.toFixed(2)}s</strong> (${m.timestamp_str})</div>
            <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 0.2rem;">Detections: ${m.boxes.length} objects bounded</div>
          </div>
        </div>
      `;
    });

    resultsGrid.innerHTML = cards;
  } catch (err) {
    resultsGrid.innerHTML = `<div style="grid-column: 1/-1; color: var(--accent-danger); text-align: center; padding: 2rem;">Search query failed: ${err.message}</div>`;
  } finally {
    queryBtn.disabled = false;
  }
}

/* ==========================================================================
   STANDING QUERIES & LIVE ALERTS
   ========================================================================== */
function initAlerts() {
  const addRuleBtn = document.getElementById("add-rule-btn");
  const evalBtn = document.getElementById("eval-alerts-btn");

  if (addRuleBtn) {
    addRuleBtn.addEventListener("click", async () => {
      const name = document.getElementById("rule-name-input").value.trim();
      const condition = document.getElementById("rule-cond-select").value;
      const threshold = parseFloat(document.getElementById("rule-thresh-slider").value);

      if (!name) {
        showToast("Please provide a rule name", "warning");
        return;
      }

      await fetch("/api/alerts/rules", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: name, condition: condition, threshold: threshold, camera: "all" }),
      });

      showToast(`Rule '${name}' activated!`, "success");
      loadAlertsData();
    });
  }

  if (evalBtn) {
    evalBtn.addEventListener("click", () => evaluateStandingAlerts());
  }

  const threshSlider = document.getElementById("rule-thresh-slider");
  const threshVal = document.getElementById("rule-thresh-val");
  if (threshSlider && threshVal) {
    threshSlider.addEventListener("input", () => {
      threshVal.textContent = parseFloat(threshSlider.value).toFixed(2);
    });
  }
}

async function loadAlertsData() {
  const rulesList = document.getElementById("standing-rules-table-body");
  if (!rulesList) return;

  try {
    const res = await fetch("/api/alerts/rules");
    const data = await res.json();

    let rows = "";
    (data.rules || []).forEach((r) => {
      rows += `
        <tr>
          <td><strong style="color: #fff;">${r.name}</strong></td>
          <td><span class="badge-pill badge-indigo">${r.condition}</span></td>
          <td>${r.camera}</td>
          <td><code style="color: var(--accent-primary);">${r.threshold.toFixed(2)}</code></td>
          <td>
            <button class="btn btn-outline btn-sm" onclick="toggleAlertRule('${r.id}')">
              ${r.enabled ? '<span style="color: var(--accent-success);">Active</span>' : '<span style="color: var(--text-muted);">Disabled</span>'}
            </button>
          </td>
        </tr>
      `;
    });

    rulesList.innerHTML = rows;
  } catch (err) {
    console.warn("Rules load failed", err);
  }

  evaluateStandingAlerts();
}

window.toggleAlertRule = async function (ruleId) {
  await fetch(`/api/alerts/rules/${ruleId}/toggle`, { method: "POST" });
  loadAlertsData();
};

async function evaluateStandingAlerts() {
  const feed = document.getElementById("alerts-triggered-feed");
  if (!feed) return;

  try {
    const res = await fetch("/api/alerts/evaluate");
    const data = await res.json();

    if (!data.evaluated || !data.alerts || data.alerts.length === 0) {
      feed.innerHTML = `<p style="padding: 2rem; text-align: center; color: var(--text-muted);">No standing rule events currently trigger alerts.</p>`;
      return;
    }

    let cards = "";
    data.alerts.forEach((alt) => {
      cards += `
        <div style="background: var(--bg-card); border: 1px solid var(--border-card); border-radius: 12px; padding: 1rem; display: flex; gap: 1rem; align-items: center;">
          ${alt.crop ? `<img src="${alt.crop}" style="width: 100px; height: 75px; object-fit: cover; border-radius: 6px; cursor: pointer;" onclick="openImageModal('${alt.crop}', '${alt.rule_name}')" />` : ""}
          <div style="flex: 1;">
            <div style="display: flex; align-items: center; justify-content: space-between;">
              <strong style="color: #fff;">${alt.rule_name}</strong>
              <span class="badge-pill badge-amber">Score: ${(alt.score * 100).toFixed(0)}%</span>
            </div>
            <div style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.3rem;">
              Camera: <strong>${alt.camera}</strong> @ <span style="font-family: var(--font-mono);">${alt.timestamp_str}</span>
            </div>
            <div style="font-size: 0.75rem; color: var(--text-muted);">Event Type: ${alt.event_type} (${alt.label})</div>
          </div>
        </div>
      `;
    });

    feed.innerHTML = cards;
  } catch (err) {
    feed.innerHTML = `<p style="color: var(--accent-danger);">Alert eval failed.</p>`;
  }
}

/* ==========================================================================
   PRIVACY REDACTION
   ========================================================================== */
function initPrivacy() {
  const toggle = document.getElementById("privacy-main-toggle");
  if (toggle) {
    toggle.addEventListener("change", async (e) => {
      await fetch("/api/privacy/toggle", { method: "POST" });
      AppState.privacyEnabled = e.target.checked;
      updatePrivacyUI(e.target.checked);
      loadPrivacyData();
    });
  }
}

async function loadPrivacyData() {
  const toggle = document.getElementById("privacy-main-toggle");
  if (toggle) toggle.checked = AppState.privacyEnabled;

  const rawImg = document.getElementById("privacy-raw-img");
  const redImg = document.getElementById("privacy-redacted-img");
  const notice = document.getElementById("privacy-demo-notice");

  try {
    const res = await fetch("/api/privacy/comparison");
    const data = await res.json();

    if (data.available && rawImg && redImg) {
      rawImg.src = data.raw;
      redImg.src = data.redacted;
      if (notice) notice.textContent = `Live Sample from ${data.camera} @ ${data.timestamp}s`;
    }
  } catch (err) {
    console.warn("Privacy comparison load failed", err);
  }
}

/* ==========================================================================
   RESEARCH BENCHMARK & ABLATION STUDY
   ========================================================================== */
function initResearch() {
  const benchBtn = document.getElementById("btn-run-live-benchmark");
  if (benchBtn) {
    benchBtn.addEventListener("click", async () => {
      benchBtn.disabled = true;
      benchBtn.innerHTML = `<span>Measuring Hardware...</span>`;
      showToast("Executing live hardware benchmark suite...", "info");
      try {
        const res = await fetch("/api/research/benchmark/run", { method: "POST" });
        const data = await res.json();
        renderLiveBenchmarkResults(data);
        showToast(`Benchmark complete! Avg latency: ${data.average_query_latency_ms} ms`, "success");
      } catch (err) {
        showToast("Live benchmark run failed", "danger");
      } finally {
        benchBtn.disabled = false;
        benchBtn.innerHTML = `<span>⚡ Run Live Hardware Benchmark</span>`;
      }
    });
  }
  loadResearchData();
}

function renderLiveBenchmarkResults(data) {
  const box = document.getElementById("live-benchmark-results");
  if (!box) return;
  box.style.display = "block";
  box.innerHTML = `
    <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid var(--accent-primary); border-radius: var(--radius-md); padding: 1.2rem; margin-bottom: 1.5rem;">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
        <h4 style="color: var(--accent-primary); font-size: 0.95rem; margin: 0; font-weight: 600;">
          ⚡ Empirical Benchmark Results (Hardware Measured)
        </h4>
        <span class="badge-pill badge-cyan">${data.device.toUpperCase()} · ${data.os}</span>
      </div>
      <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; text-align: center;">
        <div style="background: rgba(15,23,42,0.6); padding: 0.8rem; border-radius: 6px;">
          <span style="font-size: 0.75rem; color: var(--text-muted); display: block;">Avg Latency</span>
          <strong style="font-size: 1.15rem; color: #fff;">${data.average_query_latency_ms} ms</strong>
        </div>
        <div style="background: rgba(15,23,42,0.6); padding: 0.8rem; border-radius: 6px;">
          <span style="font-size: 0.75rem; color: var(--text-muted); display: block;">Grounding Rate</span>
          <strong style="font-size: 1.15rem; color: var(--accent-success);">${(data.grounding_accuracy_evaluated * 100).toFixed(1)}%</strong>
        </div>
        <div style="background: rgba(15,23,42,0.6); padding: 0.8rem; border-radius: 6px;">
          <span style="font-size: 0.75rem; color: var(--text-muted); display: block;">Abstention on Negatives</span>
          <strong style="font-size: 1.15rem; color: var(--accent-success);">${(data.hard_negative_abstention_rate * 100).toFixed(1)}%</strong>
        </div>
        <div style="background: rgba(15,23,42,0.6); padding: 0.8rem; border-radius: 6px;">
          <span style="font-size: 0.75rem; color: var(--text-muted); display: block;">Queries Tested</span>
          <strong style="font-size: 1.15rem; color: #fff;">${data.total_queries_evaluated}</strong>
        </div>
      </div>
      <div style="margin-top: 0.8rem; font-size: 0.78rem; color: var(--text-secondary);">
        Measured with system wall-clock timing. Verified report written to <code>reports/benchmark_report.json</code>.
      </div>
    </div>
  `;
}

async function loadResearchData() {
  const tbody = document.getElementById("ablation-table-body");
  if (!tbody) return;

  try {
    const res = await fetch("/api/research/ablation");
    const data = await res.json();

    let rows = "";
    (data.matrix || []).forEach((row) => {
      const isOurs = row.id.includes("B5");
      rows += `
        <tr class="${isOurs ? "highlight-row" : ""}">
          <td>
            <strong style="color: ${isOurs ? "var(--accent-primary)" : "#fff"}">${row.id}</strong>
            <span style="font-size: 0.65rem; color: #f59e0b; display: block;">[FIXTURE]</span>
          </td>
          <td>${row.name}</td>
          <td><code style="color: var(--accent-primary); font-weight: 600;">${row.retrieval_map.toFixed(3)}</code></td>
          <td>${(row.grounding_accuracy * 100).toFixed(1)}%</td>
          <td>${row.clarify_memory_persists ? '<span class="badge-pill badge-emerald">Yes</span>' : '<span style="color: var(--text-muted);">No</span>'}</td>
          <td>${row.cross_camera_continuity ? '<span class="badge-pill badge-emerald">Yes</span>' : '<span style="color: var(--text-muted);">No</span>'}</td>
          <td>${row.query_latency_ms.toFixed(1)} ms</td>
          <td><strong style="color: #fff;">${row.speedup_vs_baseline}</strong></td>
          <td><span style="color: ${row.delta_vs_baseline.startsWith("+") ? "var(--accent-success)" : "var(--accent-danger)"}">${row.delta_vs_baseline}</span></td>
        </tr>
      `;
    });

    tbody.innerHTML = rows;

    // KPIs
    if (data.kpis) {
      document.getElementById("res-kpi-map").textContent = data.kpis.retrieval_map;
      document.getElementById("res-kpi-map-delta").textContent = data.kpis.retrieval_map_delta;
      document.getElementById("res-kpi-ground").textContent = data.kpis.grounding_accuracy;
      document.getElementById("res-kpi-ground-delta").textContent = data.kpis.grounding_accuracy_delta;
      document.getElementById("res-kpi-lat").textContent = data.kpis.query_latency;
      document.getElementById("res-kpi-lat-delta").textContent = data.kpis.query_latency_delta;
    }
  } catch (err) {
    console.error("Research ablation load failed", err);
  }
}

/* ==========================================================================
   CAMERA INGESTION & MANAGEMENT
   ========================================================================== */
function initIngestion() {
  const quickBtn = document.getElementById("ingest-quick-btn");
  const customBtn = document.getElementById("ingest-custom-btn");
  const uploadInput = document.getElementById("ingest-upload-input");

  if (quickBtn) {
    quickBtn.addEventListener("click", () => triggerQuickIndex());
  }

  if (customBtn) {
    customBtn.addEventListener("click", async () => {
      const fps = parseFloat(document.getElementById("ingest-fps-slider")?.value || "2.5");
      const sim = parseFloat(document.getElementById("ingest-sim-slider")?.value || "0.7");
      const windowSec = parseFloat(document.getElementById("ingest-window-slider")?.value || "45.0");

      const sources = [
        { name: "Gate Cam", source: "data/videos/gate_cam.mp4", time_offset: 0.0 },
        { name: "Rear Cam", source: "data/videos/rear_cam.mp4", time_offset: 0.0 },
      ];

      customBtn.disabled = true;
      customBtn.innerHTML = `<span>Ingesting Streams...</span>`;
      try {
        const res = await fetch("/api/ingest", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            sources: sources,
            sample_fps: fps,
            similarity_threshold: sim,
            max_handover_seconds: windowSec,
          }),
        });
        const data = await res.json();
        showToast(`Ingested ${data.camera_count} cameras! ${data.vehicles_count} vehicles tracked.`, "success");
        await refreshSystemStatus();
      } catch (err) {
        showToast("Ingest failed", "danger");
      } finally {
        customBtn.disabled = false;
        customBtn.innerHTML = `<span>Ingest & Run Pipeline</span>`;
      }
    });
  }

  if (uploadInput) {
    uploadInput.addEventListener("change", async (e) => {
      const files = e.target.files;
      if (!files || files.length === 0) return;

      const formData = new FormData();
      for (let i = 0; i < files.length; i++) {
        formData.append("files", files[i]);
      }

      showToast("Uploading video files...", "info");
      try {
        const res = await fetch("/api/videos/upload", { method: "POST", body: formData });
        const data = await res.json();
        showToast(`Uploaded ${data.uploaded.length} clips!`, "success");
        loadIngestData();
      } catch (err) {
        showToast("Upload failed", "danger");
      }
    });
  }

  // Sliders in ingest view
  bindSliderVal("ingest-fps-slider", "ingest-fps-val");
  bindSliderVal("ingest-sim-slider", "ingest-sim-val");
  bindSliderVal("ingest-window-slider", "ingest-window-val");
}

function bindSliderVal(sliderId, valId) {
  const slider = document.getElementById(sliderId);
  const val = document.getElementById(valId);
  if (slider && val) {
    slider.addEventListener("input", () => {
      val.textContent = slider.value;
    });
  }
}

async function triggerQuickIndex() {
  const btn = document.getElementById("ingest-quick-btn");
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = `<span>Indexing Streams...</span>`;
  }
  showToast("Quick indexing Gate Cam & Rear Cam...", "info");

  try {
    const res = await fetch("/api/ingest/quick", { method: "POST" });
    const data = await res.json();
    showToast(`Quick-Indexed ${data.camera_count} cameras and ${data.vehicles_count} vehicles!`, "success");
    await refreshSystemStatus();
    if (AppState.currentView === "reid") loadReIDData();
  } catch (err) {
    showToast("Quick index failed: " + err.message, "danger");
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = `<span>Quick-Index Gate Cam & Rear Cam</span>`;
    }
  }
}

async function loadIngestData() {
  const list = document.getElementById("available-videos-list");
  const searchSelect = document.getElementById("search-clip-select");
  if (!list) return;

  try {
    const res = await fetch("/api/videos");
    const data = await res.json();
    AppState.activeVideos = data.videos || [];

    let rows = "";
    let selectOptions = "";
    (data.videos || []).forEach((v) => {
      selectOptions += `<option value="${v.path}">${v.name} (${v.size_mb} MB)</option>`;
      rows += `
        <div style="background: var(--bg-card); border: 1px solid var(--border-card); border-radius: 8px; padding: 0.8rem 1.2rem; display: flex; align-items: center; justify-content: space-between;">
          <div>
            <strong style="color: #fff;">${v.name}</strong>
            <span style="font-size: 0.75rem; color: var(--text-muted); margin-left: 0.8rem;">${v.size_mb} MB</span>
          </div>
          <button class="btn btn-outline btn-sm" onclick="openVideoPlayer('${v.url}', '${v.name}')">Preview Feed</button>
        </div>
      `;
    });

    list.innerHTML = rows;
    if (searchSelect) searchSelect.innerHTML = selectOptions;
  } catch (err) {
    console.warn("Video list load failed", err);
  }
}

window.openVideoPlayer = function (url, name) {
  const modal = document.getElementById("video-modal");
  const video = document.getElementById("modal-video-player");
  const title = document.getElementById("modal-video-title");
  if (modal && video) {
    video.src = url;
    if (title) title.textContent = name;
    modal.classList.add("active");
  }
};

/* ==========================================================================
   MODALS
   ========================================================================== */
function initModals() {
  const imageModal = document.getElementById("image-modal");
  const videoModal = document.getElementById("video-modal");

  document.querySelectorAll(".modal-close-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (imageModal) imageModal.classList.remove("active");
      if (videoModal) {
        videoModal.classList.remove("active");
        const v = document.getElementById("modal-video-player");
        if (v) v.pause();
      }
    });
  });

  [imageModal, videoModal].forEach((m) => {
    if (m) {
      m.addEventListener("click", (e) => {
        if (e.target === m) {
          m.classList.remove("active");
          const v = document.getElementById("modal-video-player");
          if (v) v.pause();
        }
      });
    }
  });
}
