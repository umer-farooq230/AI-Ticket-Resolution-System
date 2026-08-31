// app.js — no framework, no build step. Talks to the FastAPI backend on
// the same origin (the API mounts this folder as static files), so all
// fetch() calls below use relative paths.

const API = "";
const TOKEN_KEY = "admin_token";

// ---------------------------------------------------------------- utils --

function fmtPct(n) {
  if (n === null || n === undefined) return "—";
  return Math.round(n * 100) + "%";
}

function scoreChipClass(n) {
  if (n === null || n === undefined) return "chip-neutral";
  if (n >= 0.75) return "chip-good";
  if (n >= 0.5) return "chip-mid";
  return "chip-low";
}

function adminHeaders() {
  const token = localStorage.getItem(TOKEN_KEY);
  return token ? { "Authorization": "Bearer " + token } : {};
}

async function api(path, options = {}) {
  const res = await fetch(API + path, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...adminHeaders(),
      ...(options.headers || {}),
    },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch (_) { /* no json body */ }
    const err = new Error(detail);
    err.status = res.status;
    throw err;
  }
  if (res.status === 204) return null;
  return res.json();
}

// ---------------------------------------------------------- tab switching --

function initTabs() {
  document.querySelectorAll("#main-tabs .tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#main-tabs .tab-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const target = btn.dataset.view;
      document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
      document.getElementById("view-" + target).classList.add("active");
      if (target === "admin") refreshAdminView();
    });
  });

  document.querySelectorAll("#admin-subtabs .subtab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#admin-subtabs .subtab-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const target = btn.dataset.subview;
      document.querySelectorAll(".subview").forEach((v) => v.classList.remove("active"));
      document.getElementById("subview-" + target).classList.add("active");
      if (target === "pending") loadPending();
      if (target === "resolved") loadResolved();
    });
  });
}

// -------------------------------------------------------------- user view --

function initTicketForm() {
  const form = document.getElementById("ticket-form");
  const resultBox = document.getElementById("ticket-result");
  const submitBtn = document.getElementById("ticket-submit-btn");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const subject = document.getElementById("ticket-subject").value.trim();
    const body = document.getElementById("ticket-body").value.trim();
    const user_email = document.getElementById("ticket-email").value.trim();

    submitBtn.disabled = true;
    submitBtn.textContent = "Submitting...";
    resultBox.classList.add("hidden");

    try {
      const data = await api("/tickets", {
        method: "POST",
        body: JSON.stringify({ subject, body, user_email }),
      });

      resultBox.classList.remove("hidden", "success", "pending", "error");
      resultBox.classList.add(data.decision === "auto_sent" ? "success" : "pending");

      const strong = document.createElement("strong");
      strong.textContent = data.decision === "auto_sent"
        ? "Resolved — here's your answer:"
        : "Received — under review";
      resultBox.replaceChildren(strong);

      const msg = document.createElement("div");
      msg.textContent = data.message + ` (reference #${data.ticket_reference})`;
      resultBox.appendChild(msg);

      if (data.answer) {
        const ans = document.createElement("div");
        ans.style.marginTop = "10px";
        ans.style.whiteSpace = "pre-wrap";
        ans.textContent = data.answer;
        resultBox.appendChild(ans);
      }

      form.reset();
    } catch (err) {
      resultBox.classList.remove("hidden", "success", "pending");
      resultBox.classList.add("error");
      resultBox.textContent = "Something went wrong: " + err.message;
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = "Submit ticket";
    }
  });
}

function initStatusForm() {
  const form = document.getElementById("status-form");
  const resultBox = document.getElementById("status-result");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const id = document.getElementById("status-ticket-id").value.trim();
    resultBox.classList.add("hidden");

    try {
      const ticket = await api(`/tickets/${id}`);
      resultBox.classList.remove("hidden", "success", "error");
      resultBox.classList.add("success");

      const strong = document.createElement("strong");
      strong.textContent = ticket.subject;
      resultBox.replaceChildren(strong);

      const ans = document.createElement("div");
      ans.style.whiteSpace = "pre-wrap";
      ans.textContent = ticket.answer;
      resultBox.appendChild(ans);
    } catch (err) {
      resultBox.classList.remove("hidden", "success");
      resultBox.classList.add("error");
      resultBox.textContent = err.status === 404
        ? "No ticket found with that reference number."
        : "Something went wrong: " + err.message;
    }
  });
}

// ------------------------------------------------------------- admin view --

function initAdminLogin() {
  const form = document.getElementById("login-form");
  const errorBox = document.getElementById("login-error");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const password = document.getElementById("admin-password").value;
    errorBox.classList.add("hidden");

    try {
      const data = await api("/admin/login", {
        method: "POST",
        body: JSON.stringify({ password }),
      });
      localStorage.setItem(TOKEN_KEY, data.token);
      document.getElementById("admin-password").value = "";
      showDashboard();
    } catch (err) {
      errorBox.textContent = err.status === 401 ? "Incorrect password." : err.message;
      errorBox.classList.remove("hidden");
    }
  });

  document.getElementById("logout-btn").addEventListener("click", () => {
    localStorage.removeItem(TOKEN_KEY);
    showLogin();
  });

  document.getElementById("refresh-btn").addEventListener("click", refreshAdminView);
}

function showLogin() {
  document.getElementById("admin-login").classList.remove("hidden");
  document.getElementById("admin-dashboard").classList.add("hidden");
}

function showDashboard() {
  document.getElementById("admin-login").classList.add("hidden");
  document.getElementById("admin-dashboard").classList.remove("hidden");
  refreshAdminView();
}

function refreshAdminView() {
  if (!localStorage.getItem(TOKEN_KEY)) {
    showLogin();
    return;
  }
  loadPendingCount();
  const activeSubview = document.querySelector("#admin-subtabs .subtab-btn.active")?.dataset.subview || "pending";
  if (activeSubview === "pending") loadPending(); else loadResolved();
}

async function loadPendingCount() {
  try {
    const data = await api("/admin/pending/count?status=pending");
    document.getElementById("pending-count").textContent = data.count;
  } catch (err) {
    if (err.status === 401) { localStorage.removeItem(TOKEN_KEY); showLogin(); }
  }
}

async function loadPending() {
  const list = document.getElementById("pending-list");
  const empty = document.getElementById("pending-empty");

  try {
    const items = await api("/admin/pending?status=pending");
    list.replaceChildren();
    empty.classList.toggle("hidden", items.length > 0);
    items.forEach((item) => list.appendChild(renderPendingCard(item)));
  } catch (err) {
    if (err.status === 401) { localStorage.removeItem(TOKEN_KEY); showLogin(); return; }
    list.replaceChildren();
    const p = document.createElement("p");
    p.className = "error-text";
    p.textContent = "Failed to load pending reviews: " + err.message;
    list.appendChild(p);
  }
}

function renderPendingCard(item) {
  const tpl = document.getElementById("tpl-pending-card");
  const node = tpl.content.cloneNode(true);

  node.querySelector(".ticket-subject").textContent = item.subject || "(no subject)";
  node.querySelector(".badge-id").textContent = "#" + item.id;
  node.querySelector(".ticket-body").textContent = item.body;

  const confChip = node.querySelector(".chip-confidence");
  confChip.textContent = "Confidence " + fmtPct(item.composite_confidence);
  confChip.classList.add(scoreChipClass(item.composite_confidence));

  const retrChip = node.querySelector(".chip-retrieval");
  retrChip.textContent = "Retrieval " + fmtPct(item.composite_retrieval_score);
  retrChip.classList.add(scoreChipClass(item.composite_retrieval_score));

  const supChip = node.querySelector(".chip-support");
  supChip.textContent = (item.supporting_match_count ?? 0) + " supporting match"
    + ((item.supporting_match_count === 1) ? "" : "es");
  supChip.classList.add(item.supporting_match_count >= 2 ? "chip-good" : "chip-mid");

  const riskRow = node.querySelector(".chip-row-risk");
  (item.risk_flags || []).forEach((flag) => {
    const chip = document.createElement("span");
    chip.className = "chip chip-risk";
    chip.textContent = flag.replace(/_/g, " ");
    riskRow.appendChild(chip);
  });

  node.querySelector(".chip-type").textContent = item.suggested_type || "—";
  node.querySelector(".chip-queue").textContent = item.suggested_queue || "—";
  node.querySelector(".chip-priority").textContent = item.suggested_priority || "—";

  const draftArea = node.querySelector(".draft-answer");
  draftArea.value = item.draft_answer || "";

  const clarifyBox = node.querySelector(".clarifying");
  if (item.clarifying_question) {
    clarifyBox.classList.remove("hidden");
    node.querySelector(".clarifying-text").textContent = item.clarifying_question;
  }

  const gateBox = node.querySelector(".gate-failures");
  if (item.gate_failures && item.gate_failures.length) {
    gateBox.classList.remove("hidden");
    const list = gateBox.querySelector(".gate-failures-list");
    item.gate_failures.forEach((reason) => {
      const li = document.createElement("li");
      li.textContent = reason;
      list.appendChild(li);
    });
  }

  node.querySelector(".btn-approve").addEventListener("click", async (e) => {
    const btn = e.currentTarget;
    btn.disabled = true;
    btn.textContent = "Sending...";
    try {
      await api(`/admin/pending/${item.id}/approve`, {
        method: "POST",
        body: JSON.stringify({ final_answer: draftArea.value }),
      });
      loadPending();
      loadPendingCount();
    } catch (err) {
      alert("Failed to approve: " + err.message);
      btn.disabled = false;
      btn.textContent = "Approve & send";
    }
  });

  node.querySelector(".btn-reject").addEventListener("click", async (e) => {
    const note = prompt("Optional note for why this is being rejected:", "");
    if (note === null) return;
    const btn = e.currentTarget;
    btn.disabled = true;
    try {
      await api(`/admin/pending/${item.id}/reject`, {
        method: "POST",
        body: JSON.stringify({ note }),
      });
      loadPending();
      loadPendingCount();
    } catch (err) {
      alert("Failed to reject: " + err.message);
      btn.disabled = false;
    }
  });

  return node;
}

async function loadResolved() {
  const list = document.getElementById("resolved-list");
  const empty = document.getElementById("resolved-empty");
  const source = document.getElementById("resolved-source-filter").value;
  const qs = source ? `?source=${encodeURIComponent(source)}` : "";

  try {
    const items = await api("/admin/tickets" + qs);
    list.replaceChildren();
    empty.classList.toggle("hidden", items.length > 0);
    items.forEach((item) => list.appendChild(renderResolvedRow(item)));
  } catch (err) {
    if (err.status === 401) { localStorage.removeItem(TOKEN_KEY); showLogin(); return; }
    list.replaceChildren();
    const p = document.createElement("p");
    p.className = "error-text";
    p.textContent = "Failed to load tickets: " + err.message;
    list.appendChild(p);
  }
}

function renderResolvedRow(item) {
  const tpl = document.getElementById("tpl-resolved-row");
  const node = tpl.content.cloneNode(true);

  node.querySelector(".badge-id").textContent = "#" + item.id;
  node.querySelector(".resolved-subject").textContent = item.subject || "(no subject)";

  const sourceChip = node.querySelector(".chip-source");
  sourceChip.textContent = item.source;
  sourceChip.classList.add("chip-source-" + item.source);

  node.querySelector(".meta-queue").textContent = item.queue || "—";
  node.querySelector(".meta-priority").textContent = "priority: " + (item.priority || "—");
  node.querySelector(".meta-confidence").textContent = item.resolution_confidence != null
    ? "confidence: " + fmtPct(item.resolution_confidence)
    : "confidence: —";

  node.querySelector(".resolved-answer").textContent = item.answer;

  const detailBox = node.querySelector(".resolved-detail");
  const viewBtn = node.querySelector(".btn-view");
  let loaded = false;

  viewBtn.addEventListener("click", async () => {
    const showing = !detailBox.classList.contains("hidden");
    if (showing) {
      detailBox.classList.add("hidden");
      viewBtn.textContent = "View details";
      return;
    }
    if (!loaded) {
      viewBtn.disabled = true;
      viewBtn.textContent = "Loading...";
      try {
        const detail = await api(`/admin/tickets/${item.id}`);
        detailBox.replaceChildren(...buildDetailRows(detail));
        loaded = true;
      } catch (err) {
        detailBox.replaceChildren();
        const p = document.createElement("p");
        p.className = "error-text";
        p.textContent = "Failed to load details: " + err.message;
        detailBox.appendChild(p);
      }
      viewBtn.disabled = false;
    }
    detailBox.classList.remove("hidden");
    viewBtn.textContent = "Hide details";
  });

  node.querySelector(".btn-delete").addEventListener("click", async (e) => {
    if (!confirm(`Delete ticket #${item.id}? This removes it from the knowledge base permanently.`)) return;
    const btn = e.currentTarget;
    btn.disabled = true;
    btn.textContent = "Deleting...";
    try {
      await api(`/admin/tickets/${item.id}`, { method: "DELETE" });
      btn.closest(".resolved-row").remove();
      const list = document.getElementById("resolved-list");
      if (!list.children.length) document.getElementById("resolved-empty").classList.remove("hidden");
    } catch (err) {
      alert("Failed to delete: " + err.message);
      btn.disabled = false;
      btn.textContent = "Delete";
    }
  });

  return node;
}

function buildDetailRows(detail) {
  const fields = [
    ["Full message", detail.body],
    ["Full answer", detail.answer],
    ["Type", detail.type],
    ["Queue", detail.queue],
    ["Priority", detail.priority],
    ["Source", detail.source],
    ["Resolution confidence", detail.resolution_confidence != null ? fmtPct(detail.resolution_confidence) : "—"],
    ["Retrieval score", detail.resolution_retrieval_score != null ? fmtPct(detail.resolution_retrieval_score) : "—"],
    ["Supporting matches", detail.supporting_match_count ?? "—"],
    ["Risk flags", (detail.risk_flags || []).length ? detail.risk_flags.join(", ") : "none"],
    ["Version", detail.version || "—"],
    ["Language", detail.language || "—"],
  ];
  return fields.map(([label, value]) => {
    const row = document.createElement("div");
    row.className = "detail-row";
    const l = document.createElement("span");
    l.className = "detail-label";
    l.textContent = label;
    const v = document.createElement("span");
    v.className = "detail-value";
    v.textContent = value;
    row.append(l, v);
    return row;
  });
}

// ---------------------------------------------------------------- init --

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initTicketForm();
  initStatusForm();
  initAdminLogin();
  document.getElementById("resolved-source-filter").addEventListener("change", loadResolved);

  if (localStorage.getItem(TOKEN_KEY)) {
    showDashboard();
  } else {
    showLogin();
  }
});