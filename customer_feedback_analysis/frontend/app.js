/* Feedback Analysis — frontend logic (vanilla JS, no build step) */
(() => {
  "use strict";

  // Same origin when served by FastAPI; localhost:8000 if the file is opened directly.
  const API = location.protocol.startsWith("http") ? "" : "http://127.0.0.1:8000";
  const HISTORY_KEY = "feedbackAnalysis.history.v1";
  const THEME_KEY = "feedbackAnalysis.theme";
  const MAX_BATCH = 500;
  const ORDER = ["positive", "neutral", "negative"];
  const TINT = { positive: "--pos-tint", neutral: "--neu-tint", negative: "--neg-tint" };

  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pct = (x, d = 0) => `${(x * 100).toFixed(d)}%`;
  const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

  const store = {
    get(k, d) { try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : d; } catch { return d; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* unavailable */ } },
  };

  // ---------------------------------------------------------------- theme
  const themeBtn = $("#themeToggle");
  function applyTheme(t) {
    document.documentElement.setAttribute("data-theme", t);
    themeBtn.textContent = t === "dark" ? "light" : "dark";
  }
  applyTheme(store.get(THEME_KEY, matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));
  themeBtn.addEventListener("click", () => {
    const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
    applyTheme(next);
    store.set(THEME_KEY, next);
    if (current) renderTokens(current); // tints are read from CSS vars
  });

  // ---------------------------------------------------------------- toast
  let toastTimer;
  function toast(msg, isError = false) {
    const t = $("#toast");
    t.textContent = msg;
    t.classList.toggle("error", isError);
    t.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => t.classList.remove("show"), 3500);
  }

  // ---------------------------------------------------------------- api
  async function api(path, body) {
    const res = await fetch(API + path, body
      ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
      : undefined);
    if (!res.ok) {
      let detail = `${res.status} ${res.statusText}`;
      try {
        const j = await res.json();
        detail = Array.isArray(j.detail) ? j.detail.map((d) => d.msg).join("; ") : j.detail || detail;
      } catch { /* not JSON */ }
      throw new Error(detail);
    }
    return res.json();
  }

  async function checkHealth() {
    const el = $("#apiStatus");
    try {
      const m = await api("/api/model");
      el.className = "api ok";
      $("span", el).textContent = "api online";
      $("#modelMeta").textContent =
        `${m.model_type} · ${m.vocabulary_size.toLocaleString()} terms · ${m.ngram_range.join("–")}-grams`;
    } catch {
      el.className = "api err";
      $("span", el).textContent = "api offline";
      $("#modelMeta").textContent = "model unavailable";
    }
  }

  // ---------------------------------------------------------------- tabs
  function showView(name) {
    $$(".tab").forEach((t) => {
      const on = t.dataset.view === name;
      t.classList.toggle("is-active", on);
      t.setAttribute("aria-selected", on);
    });
    $$(".view").forEach((v) => (v.hidden = v.id !== `view-${name}`));
    if (name === "history") renderHistory();
  }
  $$(".tab").forEach((t) => t.addEventListener("click", () => showView(t.dataset.view)));

  function setLoading(btn, on) {
    btn.disabled = on;
    btn.classList.toggle("loading", on);
  }

  function renderStack(stackEl, legendEl, values, total) {
    stackEl.innerHTML = ORDER.filter((c) => values[c] > 0)
      .map((c) => `<span class="bg-${c}" style="width:${(values[c] / total) * 100}%" title="${c}"></span>`).join("");
    legendEl.innerHTML = ORDER.map((c) => `
      <div class="is-${c}"><dt>${c}</dt><dd>${total === 1 ? pct(values[c] || 0, 1) : `${values[c] || 0} · ${pct((values[c] || 0) / total, 1)}`}</dd></div>`).join("");
  }

  // ================================================================ SINGLE
  const input = $("#feedbackInput");
  let current = null;
  const updateCount = () => ($("#charCount").textContent = `${input.value.length} / 5000`);
  input.addEventListener("input", updateCount);
  input.addEventListener("keydown", (e) => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") analyze(); });
  $$("[data-example]").forEach((b) => b.addEventListener("click", () => {
    input.value = b.dataset.example; updateCount(); analyze();
  }));
  $("#clearBtn").addEventListener("click", () => {
    input.value = ""; updateCount(); current = null;
    $("#result").hidden = true; $("#resultEmpty").hidden = false; input.focus();
  });
  $("#analyzeBtn").addEventListener("click", analyze);

  async function analyze() {
    const text = input.value.trim();
    if (!text) { toast("Paste a review first."); input.focus(); return; }
    const btn = $("#analyzeBtn");
    setLoading(btn, true);
    try {
      const r = await api("/api/predict", { text });
      renderResult(r);
      addHistory(r);
    } catch (e) {
      toast(`Analysis failed: ${e.message}`, true);
    } finally {
      setLoading(btn, false);
    }
  }

  function renderResult(r) {
    current = r;
    $("#resultEmpty").hidden = true;
    $("#result").hidden = false;

    const word = $("#verdictWord");
    word.textContent = r.sentiment;
    word.className = `verdict-word is-${r.sentiment}`;
    $("#verdictConf").textContent = pct(r.confidence, 1);
    renderStack($("#probStack"), $("#probLegend"), r.probabilities, 1);
    $("#lowSignal").hidden = !r.low_signal;
    $("#towardClass").textContent = r.sentiment;
    renderTokens(r);

    const terms = [...r.supporting_terms, ...r.opposing_terms]
      .sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight)).slice(0, 10);
    const max = Math.max(...terms.map((t) => Math.abs(t.weight)), 1e-9);
    $("#termsBody").innerHTML = terms.length
      ? terms.map((t) => `
        <tr>
          <td>${esc(t.term)}</td>
          <td class="num">${t.weight > 0 ? "+" : ""}${t.weight.toFixed(3)}</td>
          <td><div class="effect"><span class="${t.weight > 0 ? "up" : "down"}" style="width:${(Math.abs(t.weight) / max) * 50}%"></span></div></td>
        </tr>`).join("")
      : `<tr><td colspan="3" class="meta">No known terms.</td></tr>`;
  }

  function renderTokens(r) {
    const tint = cssVar(TINT[r.sentiment] || "--pos-tint");
    const max = Math.max(...(r.tokens || []).map((t) => Math.abs(t.weight || 0)), 1e-9);
    $("#tokens").innerHTML = (r.tokens || []).length
      ? r.tokens.map((t) => {
        if (t.weight === null) return `<span class="oov">${esc(t.token)}</span>`;
        if (t.weight < 0) return `<span class="against" title="${t.weight}">${esc(t.token)}</span>`;
        const a = (0.08 + 0.5 * (t.weight / max)).toFixed(2);
        return `<span style="background:rgba(${tint},${a})" title="+${t.weight}">${esc(t.token)}</span>`;
      }).join(" ")
      : `<span class="meta">(nothing left after cleaning)</span>`;
  }

  // ================================================================ HISTORY
  let history = store.get(HISTORY_KEY, []);
  const updateHistoryCount = () => ($("#historyCount").textContent = history.length);
  updateHistoryCount();

  function addHistory(r) {
    history = [{ ...r, at: Date.now() }, ...history.filter((h) => h.text !== r.text)].slice(0, 50);
    store.set(HISTORY_KEY, history);
    updateHistoryCount();
  }

  function renderHistory() {
    const body = $("#historyList");
    if (!history.length) {
      body.innerHTML = `<tr><td class="meta">Nothing yet. Results from the Single tab are listed here.</td></tr>`;
      return;
    }
    body.innerHTML = history.map((h, i) => `
      <tr data-i="${i}" tabindex="0">
        <td><span class="sent is-${h.sentiment}">${esc(h.sentiment)}</span></td>
        <td class="num">${pct(h.confidence)}</td>
        <td class="text">${esc(h.text)}</td>
        <td class="when">${new Date(h.at).toLocaleString([], { dateStyle: "short", timeStyle: "short" })}</td>
      </tr>`).join("");
    $$("tr[data-i]", body).forEach((tr) => {
      const open = () => {
        const h = history[+tr.dataset.i];
        input.value = h.text; updateCount(); renderResult(h); showView("single");
      };
      tr.addEventListener("click", open);
      tr.addEventListener("keydown", (e) => e.key === "Enter" && open());
    });
  }
  $("#clearHistory").addEventListener("click", () => {
    history = []; store.set(HISTORY_KEY, history); updateHistoryCount(); renderHistory();
  });

  // ================================================================ BATCH
  const batchInput = $("#batchInput");
  let csvRows = null;
  let lastBatch = null;
  let filter = "all";

  const batchLines = () => batchInput.value.split(/\r?\n/).map((s) => s.trim()).filter(Boolean);
  const updateBatchCount = () => {
    const n = batchLines().length;
    $("#batchCount").textContent = `${n} review${n === 1 ? "" : "s"}${n > MAX_BATCH ? ` (first ${MAX_BATCH} used)` : ""}`;
  };
  batchInput.addEventListener("input", updateBatchCount);

  $("#batchSample").addEventListener("click", () => {
    batchInput.value = [
      "Love my new Kindle, the screen is easy on the eyes and battery lasts weeks.",
      "The Echo stopped responding after a week. Really disappointed.",
      "Bought this as a gift for my mom.",
      "Great value for the price, my kids use it every day.",
      "Sound quality is poor and the device keeps disconnecting from wifi.",
      "Setup was quick and easy. Alexa understands me perfectly.",
      "It is a tablet. It does tablet things.",
      "Worst purchase ever, returned it the next day.",
      "Perfect for reading at night, very lightweight.",
      "Fire TV stick works fine but the remote feels cheap.",
    ].join("\n");
    csvRows = null;
    $("#columnPick").hidden = true;
    $("#fileMeta").textContent = "";
    updateBatchCount();
  });

  // file handling
  const dz = $("#dropzone");
  const fileInput = $("#fileInput");
  $("#browseBtn").addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => fileInput.files[0] && loadFile(fileInput.files[0]));
  ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", (e) => e.dataTransfer.files[0] && loadFile(e.dataTransfer.files[0]));

  async function loadFile(file) {
    if (file.size > 50 * 1024 * 1024) { toast("File is larger than 50 MB.", true); return; }
    const text = await file.text();
    if (/\.txt$/i.test(file.name)) {
      csvRows = null;
      $("#columnPick").hidden = true;
      batchInput.value = text;
      $("#fileMeta").textContent = file.name;
      updateBatchCount();
      return;
    }
    const rows = parseCSV(text);
    if (rows.length < 2) { toast("That CSV looks empty.", true); return; }
    const header = rows[0];
    csvRows = rows.slice(1).filter((r) => r.some((c) => c.trim()));
    const sel = $("#columnSelect");
    sel.innerHTML = header.map((h, i) => `<option value="${i}">${esc(h || `column ${i + 1}`)}</option>`).join("");
    sel.value = String(guessTextColumn(header, csvRows));
    $("#columnPick").hidden = false;
    $("#fileMeta").textContent = `${file.name}, ${csvRows.length.toLocaleString()} rows`;
    fillFromColumn();
  }
  $("#columnSelect").addEventListener("change", fillFromColumn);

  function fillFromColumn() {
    if (!csvRows) return;
    const col = +$("#columnSelect").value;
    batchInput.value = csvRows.map((r) => (r[col] || "").replace(/\s*\r?\n\s*/g, " ").trim()).filter(Boolean).join("\n");
    updateBatchCount();
  }

  function guessTextColumn(header, rows) {
    const byName = header.findIndex((h) => /review.*text|feedback|comment|review|text|body|message/i.test(h));
    if (byName >= 0) return byName;
    let best = 0, bestLen = -1;
    const sample = rows.slice(0, 50);
    header.forEach((_, i) => {
      const avg = sample.reduce((s, r) => s + (r[i] || "").length, 0) / (sample.length || 1);
      if (avg > bestLen) { bestLen = avg; best = i; }
    });
    return best;
  }

  // CSV parser: quoted fields, escaped quotes, embedded newlines
  function parseCSV(str) {
    const rows = [];
    let row = [], field = "", q = false;
    for (let i = 0; i < str.length; i++) {
      const c = str[i];
      if (q) {
        if (c === '"') { if (str[i + 1] === '"') { field += '"'; i++; } else q = false; }
        else field += c;
      } else if (c === '"') q = true;
      else if (c === ",") { row.push(field); field = ""; }
      else if (c === "\n" || c === "\r") {
        if (c === "\r" && str[i + 1] === "\n") i++;
        row.push(field); rows.push(row); row = []; field = "";
      } else field += c;
    }
    if (field || row.length) { row.push(field); rows.push(row); }
    return rows;
  }

  $("#batchBtn").addEventListener("click", async () => {
    const texts = batchLines().slice(0, MAX_BATCH);
    if (!texts.length) { toast("Add at least one review, one per line."); batchInput.focus(); return; }
    const btn = $("#batchBtn");
    setLoading(btn, true);
    try {
      lastBatch = await api("/api/predict/batch", { texts });
      renderBatch(lastBatch);
      $("#batchResults").scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (e) {
      toast(`Batch failed: ${e.message}`, true);
    } finally {
      setLoading(btn, false);
    }
  });

  function renderBatch(b) {
    $("#batchResults").hidden = false;
    const s = b.summary;
    $("#summaryLine").innerHTML =
      `<b>${b.count}</b> reviews analyzed in ${b.elapsed_ms} ms. ` +
      `Mean confidence <b>${pct(b.average_confidence, 1)}</b>.`;
    renderStack($("#batchStack"), $("#batchLegend"), s, b.count);

    $("#drivers").innerHTML = ORDER.map((c) => {
      const list = (b.top_terms[c] || []).slice(0, 8);
      return `<div>
        <h3><span class="is-${c}">${c}</span><span class="meta">${s[c] || 0} reviews</span></h3>
        ${list.length
          ? `<ol>${list.map((t) => `<li><span>${esc(t.term)}</span><span>${t.weight.toFixed(2)}</span></li>`).join("")}</ol>`
          : `<p class="none">No reviews.</p>`}
      </div>`;
    }).join("");

    filter = "all";
    $("#tableSearch").value = "";
    $$("#filterSeg button").forEach((x) => x.classList.toggle("is-active", x.dataset.filter === "all"));
    renderTable();
  }

  $$("#filterSeg button").forEach((b) => b.addEventListener("click", () => {
    filter = b.dataset.filter;
    $$("#filterSeg button").forEach((x) => x.classList.toggle("is-active", x === b));
    renderTable();
  }));
  $("#tableSearch").addEventListener("input", renderTable);

  function renderTable() {
    if (!lastBatch) return;
    const q = $("#tableSearch").value.trim().toLowerCase();
    const rows = lastBatch.results
      .map((r, i) => ({ ...r, i: i + 1 }))
      .filter((r) => (filter === "all" || r.sentiment === filter) && (!q || r.text.toLowerCase().includes(q)));
    $("#resultsBody").innerHTML = rows.length
      ? rows.map((r) => `
        <tr>
          <td class="num">${r.i}</td>
          <td>${esc(r.text)}</td>
          <td><span class="sent is-${r.sentiment}">${r.sentiment}</span></td>
          <td class="num">${pct(r.confidence, 1)}</td>
        </tr>`).join("")
      : `<tr><td colspan="4" class="meta">No matching reviews.</td></tr>`;
  }

  $("#exportBtn").addEventListener("click", () => {
    if (!lastBatch) return;
    const q = (v) => `"${String(v).replace(/"/g, '""')}"`;
    const head = ["text", "sentiment", "confidence", "p_positive", "p_neutral", "p_negative"];
    const lines = lastBatch.results.map((r) => [
      q(r.text), r.sentiment, r.confidence,
      r.probabilities.positive ?? "", r.probabilities.neutral ?? "", r.probabilities.negative ?? "",
    ].join(","));
    const blob = new Blob(["﻿" + [head.join(","), ...lines].join("\n")], { type: "text/csv;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `feedback_sentiment_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
  });

  // ---------------------------------------------------------------- init
  updateCount();
  updateBatchCount();
  checkHealth();
  setInterval(checkHealth, 30000);
})();
