(() => {
  "use strict";

  const $ = (s) => document.querySelector(s);
  const $$ = (s) => document.querySelectorAll(s);
  const HISTORY_KEY = "spamshield.history";
  const THEME_KEY = "spamshield.theme";

  const SAMPLES = {
    spam1: "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.",
    spam2: "Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free! Call The Mobile Update Co FREE on 08002986030",
    ham1: "Hey, are we still on for coffee tomorrow at 10? Let me know if you want to move it to the afternoon instead.",
    ham2: "Hi team, just a reminder that the project review meeting has moved to Thursday at 3pm. Please bring your updated slides.",
  };

  // ---------- helpers ----------
  const store = {
    get(k, d) { try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : d; } catch { return d; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* ignore */ } },
  };

  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pct = (x, d = 1) => `${(x * 100).toFixed(d)}%`;

  let toastTimer;
  function toast(msg, isError = false) {
    const t = $("#toast");
    t.textContent = msg;
    t.className = `toast show${isError ? " error" : ""}`;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => (t.className = "toast"), 2800);
  }

  async function api(path, opts = {}) {
    const res = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...opts,
    });
    if (!res.ok) {
      let detail = res.statusText;
      try { const j = await res.json(); detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch { /* ignore */ }
      throw new Error(detail || `Request failed (${res.status})`);
    }
    return res.json();
  }

  function setLoading(btn, on) {
    btn.disabled = on;
    btn.classList.toggle("loading", on);
  }

  function timeAgo(ts) {
    const s = Math.round((Date.now() - ts) / 1000);
    if (s < 60) return "just now";
    if (s < 3600) return `${Math.floor(s / 60)}m ago`;
    if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
    return new Date(ts).toLocaleDateString();
  }

  // ---------- theme ----------
  const theme = store.get(THEME_KEY, window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  document.documentElement.dataset.theme = theme;
  $("#themeBtn").addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    store.set(THEME_KEY, next);
  });

  // ---------- tabs ----------
  const loaders = { quarantine: loadQuarantine, model: loadModel };
  $$(".tab").forEach((tab) =>
    tab.addEventListener("click", () => {
      $$(".tab").forEach((t) => t.classList.toggle("active", t === tab));
      $$(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${tab.dataset.view}`));
      loaders[tab.dataset.view]?.();
    })
  );

  // ---------- health ----------
  async function checkHealth() {
    const el = $("#apiStatus");
    try {
      const h = await api("/api/health");
      el.className = "status ok";
      el.querySelector(".txt").textContent = `${h.model} online`;
    } catch {
      el.className = "status down";
      el.querySelector(".txt").textContent = "API offline";
    }
  }

  // ---------- analyze ----------
  const input = $("#emailInput");
  const analyzeBtn = $("#analyzeBtn");

  input.addEventListener("input", () => ($("#charCount").textContent = `${input.value.length.toLocaleString()} chars`));
  input.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") analyze();
  });
  analyzeBtn.addEventListener("click", analyze);
  $("#clearBtn").addEventListener("click", () => {
    input.value = "";
    input.dispatchEvent(new Event("input"));
    $("#resultBody").classList.add("hidden");
    $("#resultEmpty").classList.remove("hidden");
    input.focus();
  });
  $$(".chip").forEach((c) =>
    c.addEventListener("click", () => {
      input.value = SAMPLES[c.dataset.sample];
      input.dispatchEvent(new Event("input"));
      analyze();
    })
  );

  async function analyze() {
    const text = input.value.trim();
    if (!text) { toast("Paste an email first.", true); input.focus(); return; }
    setLoading(analyzeBtn, true);
    try {
      const r = await api("/api/predict", { method: "POST", body: JSON.stringify({ text }) });
      renderResult(r);
      addHistory(text, r);
      if (r.label === "spam") refreshQCount();
    } catch (e) {
      toast(e.message, true);
    } finally {
      setLoading(analyzeBtn, false);
    }
  }

  function renderResult(r) {
    $("#resultEmpty").classList.add("hidden");
    $("#resultBody").classList.remove("hidden");
    const isSpam = r.label === "spam";

    const v = $("#verdict");
    v.className = `verdict ${r.label}`;
    $("#verdictIcon").textContent = isSpam ? "!" : "✓";
    $("#verdictLabel").textContent = isSpam ? "Spam" : "Not spam (Ham)";
    $("#verdictSub").textContent = `${pct(r.confidence)} confidence · ${r.latency_ms} ms${isSpam ? " · moved to quarantine" : ""}`;

    const p = r.spam_probability;
    const fill = $("#gaugeFill");
    fill.style.stroke = p >= 0.5 ? "var(--spam)" : "var(--ham)";
    requestAnimationFrame(() => (fill.style.strokeDashoffset = String(314.16 * (1 - p))));
    $("#gaugeVal").textContent = pct(p, 0);
    $("#meterMark").style.left = `${p * 100}%`;

    const list = $("#signals");
    if (!r.signals.length) {
      list.innerHTML = `<li><span class="none">None of these words appear in the model's vocabulary, so the verdict relies on class priors only.</span></li>`;
    } else {
      const max = Math.max(...r.signals.map((s) => Math.abs(s.weight)), 1);
      list.innerHTML = r.signals
        .map((s) => {
          const w = (Math.abs(s.weight) / max) * 50;
          const cls = s.weight >= 0 ? "spam" : "ham";
          return `<li><span class="term" title="${esc(s.term)}">${esc(s.term)}${s.count > 1 ? ` ×${s.count}` : ""}</span>
            <span class="track"><span class="bar ${cls}" style="width:${w}%"></span></span>
            <span class="w">${s.weight > 0 ? "+" : ""}${s.weight.toFixed(2)}</span></li>`;
        })
        .join("");
    }
    $("#cleanedText").textContent = r.cleaned_text || "(empty after cleaning)";
    $("#termsInfo").textContent = `· ${r.known_terms} known term${r.known_terms === 1 ? "" : "s"}`;
  }

  // ---------- history ----------
  function addHistory(text, r) {
    const h = store.get(HISTORY_KEY, []);
    h.unshift({ text, label: r.label, p: r.spam_probability, ts: Date.now() });
    store.set(HISTORY_KEY, h.slice(0, 15));
    renderHistory();
  }
  function renderHistory() {
    const h = store.get(HISTORY_KEY, []);
    const ul = $("#historyList");
    if (!h.length) { ul.innerHTML = `<li class="muted small">No checks yet.</li>`; return; }
    ul.innerHTML = h
      .map((x, i) => `<li class="item" data-i="${i}"><span class="pill ${x.label}">${x.label}</span>
        <span class="msg">${esc(x.text)}</span><span class="time">${pct(x.p, 0)} · ${timeAgo(x.ts)}</span></li>`)
      .join("");
  }
  $("#historyList").addEventListener("click", (e) => {
    const li = e.target.closest("li.item");
    if (!li) return;
    input.value = store.get(HISTORY_KEY, [])[li.dataset.i].text;
    input.dispatchEvent(new Event("input"));
    window.scrollTo({ top: 0, behavior: "smooth" });
    analyze();
  });
  $("#clearHistory").addEventListener("click", () => { store.set(HISTORY_KEY, []); renderHistory(); });

  // ---------- batch ----------
  const batchInput = $("#batchInput");
  let lastBatch = [];
  const batchLines = () => batchInput.value.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  batchInput.addEventListener("input", () => ($("#batchCount").textContent = `${batchLines().length} messages`));

  $("#fileInput").addEventListener("change", async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    let txt = await f.text();
    if (f.name.toLowerCase().endsWith(".csv")) {
      txt = txt.split(/\r?\n/).filter(Boolean).map((row) => {
        const cells = row.match(/("([^"]|"")*"|[^,]*)(,|$)/g) || [row];
        const last = cells.filter((c) => c !== "").pop() || "";
        return last.replace(/,$/, "").replace(/^"|"$/g, "").replace(/""/g, '"');
      }).join("\n");
    }
    batchInput.value = txt;
    batchInput.dispatchEvent(new Event("input"));
    e.target.value = "";
  });

  $("#batchBtn").addEventListener("click", async () => {
    const texts = batchLines();
    if (!texts.length) { toast("Add at least one message.", true); return; }
    if (texts.length > 100) { toast("Max 100 messages per batch.", true); return; }
    const btn = $("#batchBtn");
    setLoading(btn, true);
    try {
      const r = await api("/api/predict/batch", { method: "POST", body: JSON.stringify({ texts }) });
      lastBatch = r.results;
      $("#bSpam").textContent = `${r.spam} spam`;
      $("#bHam").textContent = `${r.count - r.spam} ham`;
      $("#batchBody").innerHTML = r.results
        .map((x, i) => `<tr><td class="num muted">${i + 1}</td><td class="msg">${esc(x.text)}</td>
          <td><span class="pill ${x.label}">${x.label}</span></td>
          <td class="num"><span class="mini"><i style="width:${x.spam_probability * 100}%"></i></span>${pct(x.spam_probability)}</td></tr>`)
        .join("");
      $("#batchResults").classList.remove("hidden");
      if (r.spam) refreshQCount();
    } catch (e) {
      toast(e.message, true);
    } finally {
      setLoading(btn, false);
    }
  });

  $("#exportBtn").addEventListener("click", () => {
    const q = (s) => `"${String(s).replace(/"/g, '""')}"`;
    const csv = ["text,label,spam_probability", ...lastBatch.map((r) => [q(r.text), r.label, r.spam_probability].join(","))].join("\n");
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    a.download = "spam_predictions.csv";
    a.click();
    URL.revokeObjectURL(a.href);
  });

  // ---------- quarantine ----------
  async function refreshQCount() {
    try {
      const { items } = await api("/api/quarantine?limit=1000");
      $("#qCount").textContent = items.length;
      return items;
    } catch { return []; }
  }
  async function loadQuarantine() {
    const items = await refreshQCount();
    const body = $("#qBody");
    body.innerHTML = items.length
      ? items.slice(0, 200).map((x) => {
          const p = parseFloat(x.spam_probability);
          return `<tr><td class="num muted">${esc(new Date(x.timestamp).toLocaleString())}</td>
            <td class="msg">${esc(x.email_text)}</td>
            <td class="num">${isNaN(p) ? "—" : `<span class="mini"><i style="width:${p * 100}%"></i></span>${pct(p)}`}</td></tr>`;
        }).join("")
      : `<tr><td colspan="3" class="muted">Quarantine is empty.</td></tr>`;
  }
  $("#refreshQ").addEventListener("click", loadQuarantine);
  $("#clearQ").addEventListener("click", async () => {
    if (!confirm("Delete all quarantined messages? This cannot be undone.")) return;
    try { await api("/api/quarantine", { method: "DELETE" }); toast("Quarantine cleared."); loadQuarantine(); }
    catch (e) { toast(e.message, true); }
  });

  // ---------- model ----------
  let modelLoaded = false;
  async function loadModel() {
    if (modelLoaded) return;
    $("#stats").innerHTML = `<div class="stat"><div class="k">Loading…</div><div class="v">—</div></div>`;
    try {
      const m = await api("/api/model-info");
      modelLoaded = true;
      $("#stats").innerHTML = Object.entries(m.metrics)
        .map(([k, v]) => `<div class="stat"><div class="k">${k === "f1" ? "F1 score" : k}</div><div class="v">${pct(v)}</div><div class="bar"><i style="width:${v * 100}%"></i></div></div>`)
        .join("");

      const [[tp, fn], [fp, tn]] = m.confusion_matrix.matrix;
      $("#cmSub").textContent = `Held-out test set · ${m.test_size.toLocaleString()} messages (rows = actual, columns = predicted)`;
      $("#cm").innerHTML = `
        <div></div><div class="h">Pred. spam</div><div class="h">Pred. ham</div>
        <div class="h">Actual spam</div><div class="c good"><b>${tp}</b><span>caught</span></div><div class="c bad"><b>${fn}</b><span>missed</span></div>
        <div class="h">Actual ham</div><div class="c bad"><b>${fp}</b><span>false alarm</span></div><div class="c good"><b>${tn}</b><span>correct</span></div>`;

      const total = Object.values(m.dataset).reduce((a, b) => a + b, 0);
      $("#details").innerHTML = `
        <dt>Algorithm</dt><dd>${esc(m.model)}</dd>
        <dt>Features</dt><dd>${esc(m.vectorizer)}</dd>
        <dt>Dataset</dt><dd>${total.toLocaleString()} messages</dd>
        <dt>Ham</dt><dd>${m.dataset.ham.toLocaleString()} (${pct(m.dataset.ham / total)})</dd>
        <dt>Spam</dt><dd>${m.dataset.spam.toLocaleString()} (${pct(m.dataset.spam / total)})</dd>
        <dt>Balancing</dt><dd>RandomOverSampler (train split only)</dd>
        <dt>Preprocessing</dt><dd>lowercase · strip URLs/punctuation · spaCy stop-words + lemmas</dd>`;
      $("#spamTerms").innerHTML = m.top_spam_terms.map((t) => `<span>${esc(t)}</span>`).join("");
      $("#hamTerms").innerHTML = m.top_ham_terms.map((t) => `<span>${esc(t)}</span>`).join("");
    } catch (e) {
      $("#stats").innerHTML = "";
      toast(`Could not load model info: ${e.message}`, true);
    }
  }

  // ---------- init ----------
  renderHistory();
  checkHealth();
  refreshQCount();
  setInterval(checkHealth, 30000);
  input.focus();
})();
