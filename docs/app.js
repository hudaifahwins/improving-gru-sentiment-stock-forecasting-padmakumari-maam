/* Improving GRU Sentiment Stock Forecasting — interactive dashboard (no deps). */
(() => {
  "use strict";

  /* ---------------------------------------------------------------- verified data */
  const TICKERS = {
    aapl:   { label: "AAPL", long: "Apple Inc. — US10Y daily returns" },
    samsung: { label: "005930.KS", long: "Samsung Electronics — Korean 5Y-ish window" }
  };

  /* horizon decomposition: {overlap, strict, momentum} per h with std */
  const HORIZON = {
    aapl: {
      overlap:  { h1: [0.4832, 0.0089], h3: [0.7762, 0.0118], h5: [0.7839, 0.0265] },
      strict:   { h1: [0.4832, 0.0089], h3: [0.6494, 0.0347], h5: [0.7333, 0.0526] },
      momentum: { h1: [0.4841, 0.0000], h3: [0.5086, 0.0000], h5: [0.5000, 0.0000] }
    },
    samsung: {
      overlap:  { h1: [0.4224, 0.0014], h3: [0.6538, 0.0061], h5: [0.6785, 0.0028] },
      strict:   { h1: [0.4224, 0.0014], h3: [0.6254, 0.0083], h5: [0.6422, 0.0302] },
      momentum: { h1: [0.4837, 0.0000], h3: [0.5044, 0.0000], h5: [0.5000, 0.0000] }
    }
  };

  const EQUIV = {
    aapl:   { gru: [0.4832, 0.0089], ridge: [0.4832, 0.0089], gbr: [0.4832, 0.0089], momentum: [0.4841, 0.0000] },
    samsung: { gru: [0.4224, 0.0014], ridge: [0.4224, 0.0014], gbr: [0.4224, 0.0014], momentum: [0.4837, 0.0000] }
  };

  const DIAG = {
    aapl: {
      mse: [0.4890, 0.0027], strict: [0.4607, 0.0021],
      hybrid: [0.4957, 0.0209], strictHybrid: [0.4978, 0.0419],
      logvol: [0.4822, 0.0049]
    },
    samsung: {
      mse: [0.4214, 0.0000], strict: [0.4161, 0.0000],
      hybrid: [0.4362, 0.0106], strictHybrid: [0.4359, 0.0115],
      logvol: [0.4214, 0.0000]
    }
  };

  const MAE_HORIZON = {
    aapl:   { h1: 3.103, h3: 4.273, h5: 6.910 },
    samsung: { h1: 6775, h3: 47450, h5: 103778 }
  };

  const ABLATION = {
    aapl: {
      rows: 1222,
      price: [0.4832, 0.0089], tech: [0.468, 0.011], sent: [0.5067, 0.0303],
      toneDelta: [-0.0022, 0.0154], dummyDelta: 0.003
    },
    samsung: {
      rows: 1187,
      price: [0.4224, 0.0014], tech: [0.448, 0.010], sent: [0.492, 0.021],
      toneDelta: [0.0000, 0.0000], dummyDelta: null
    }
  };

  /* full explorer rows */
  const ROWS = [
    { t:"AAPL", h:1, mode:"overlap", cfg:"price · mse", dir:0.4832, sd:0.0089, strict:0.4741, strictSd:0.0076, mae:3.103, rmse:4.382 },
    { t:"AAPL", h:3, mode:"overlap", cfg:"price · mse", dir:0.7762, sd:0.0118, strict:0.8351, strictSd:0.0114, mae:4.273, rmse:null },
    { t:"AAPL", h:5, mode:"overlap", cfg:"price · mse", dir:0.7839, sd:0.0265, strict:0.8120, strictSd:0.0286, mae:6.910, rmse:null },
    { t:"AAPL", h:1, mode:"strict non-overlap", cfg:"price · mse", dir:0.4832, sd:0.0089, strict:0.4741, strictSd:0.0076, mae:null, rmse:null },
    { t:"AAPL", h:3, mode:"strict non-overlap", cfg:"price · mse", dir:0.6494, sd:0.0347, strict:0.7333, strictSd:0.0397, mae:null, rmse:null },
    { t:"AAPL", h:5, mode:"strict non-overlap", cfg:"price · mse", dir:0.7333, sd:0.0526, strict:0.7701, strictSd:0.0569, mae:null, rmse:null },
    { t:"AAPL", h:1, mode:"momentum", cfg:"causal baseline", dir:0.4841, sd:0.0000, strict:null, strictSd:null, mae:null, rmse:null },
    { t:"AAPL", h:3, mode:"momentum", cfg:"causal baseline", dir:0.5086, sd:0.0000, strict:null, strictSd:null, mae:null, rmse:null },
    { t:"AAPL", h:5, mode:"momentum", cfg:"causal baseline", dir:0.5000, sd:0.0000, strict:null, strictSd:null, mae:null, rmse:null },
    { t:"AAPL", h:1, mode:"overlap", cfg:"price+tech · mse", dir:0.468, sd:0.011, strict:0.4459, strictSd:0.0133, mae:3.525, rmse:null },
    { t:"AAPL", h:1, mode:"overlap", cfg:"price+tech+GDELT · mse", dir:0.5067, sd:0.0303, strict:0.5061, strictSd:0.0494, mae:10.842, rmse:null },
    { t:"AAPL", h:1, mode:"diagnostics", cfg:"mse/log", dir:0.4890, sd:0.0027, strict:0.4607, strictSd:0.0021, mae:null, rmse:null },
    { t:"AAPL", h:1, mode:"diagnostics", cfg:"hybrid/log", dir:0.4957, sd:0.0209, strict:0.4978, strictSd:0.0419, mae:null, rmse:null },
    { t:"AAPL", h:1, mode:"diagnostics", cfg:"logvol", dir:0.4822, sd:0.0049, strict:null, strictSd:null, mae:null, rmse:null },
    { t:"005930.KS", h:1, mode:"overlap", cfg:"price · mse", dir:0.4224, sd:0.0014, strict:0.4207, strictSd:0.0066, mae:6775, rmse:11298 },
    { t:"005930.KS", h:3, mode:"overlap", cfg:"price · mse", dir:0.6538, sd:0.0061, strict:0.6730, strictSd:0.0052, mae:47450, rmse:null },
    { t:"005930.KS", h:5, mode:"overlap", cfg:"price · mse", dir:0.6785, sd:0.0028, strict:0.6842, strictSd:0.0025, mae:103778, rmse:null },
    { t:"005930.KS", h:1, mode:"strict non-overlap", cfg:"price · mse", dir:0.4224, sd:0.0014, strict:0.4207, strictSd:0.0066, mae:null, rmse:null },
    { t:"005930.KS", h:3, mode:"strict non-overlap", cfg:"price · mse", dir:0.6254, sd:0.0083, strict:0.6538, strictSd:0.0079, mae:null, rmse:null },
    { t:"005930.KS", h:5, mode:"strict non-overlap", cfg:"price · mse", dir:0.6422, sd:0.0302, strict:0.6508, strictSd:0.0259, mae:null, rmse:null },
    { t:"005930.KS", h:1, mode:"momentum", cfg:"causal baseline", dir:0.4837, sd:0.0000, strict:null, strictSd:null, mae:null, rmse:null },
    { t:"005930.KS", h:3, mode:"momentum", cfg:"causal baseline", dir:0.5044, sd:0.0000, strict:null, strictSd:null, mae:null, rmse:null },
    { t:"005930.KS", h:5, mode:"momentum", cfg:"causal baseline", dir:0.5000, sd:0.0000, strict:null, strictSd:null, mae:null, rmse:null },
    { t:"005930.KS", h:1, mode:"overlap", cfg:"price+tech · mse", dir:0.448, sd:0.010, strict:0.4531, strictSd:0.0173, mae:8250, rmse:null },
    { t:"005930.KS", h:1, mode:"overlap", cfg:"price+tech+GDELT · mse", dir:0.492, sd:0.021, strict:0.5103, strictSd:0.0238, mae:29470, rmse:null },
    { t:"005930.KS", h:1, mode:"diagnostics", cfg:"mse/log", dir:0.4214, sd:0.0000, strict:0.4161, strictSd:0.0000, mae:null, rmse:null },
    { t:"005930.KS", h:1, mode:"diagnostics", cfg:"hybrid/log", dir:0.4362, sd:0.0106, strict:0.4359, strictSd:0.0115, mae:null, rmse:null },
    { t:"005930.KS", h:1, mode:"diagnostics", cfg:"logvol", dir:0.4214, sd:0.0000, strict:null, strictSd:null, mae:null, rmse:null },
    { t:"AAPL", h:1, mode:"ablation", cfg:"standalone GDELT tone Δ", dir:-0.0022, sd:0.0154, strict:null, strictSd:null, mae:null, rmse:null },
    { t:"AAPL", h:1, mode:"ablation", cfg:"zero-feature dummy control Δ", dir:0.003, sd:null, strict:null, strictSd:null, mae:null, rmse:null },
    { t:"005930.KS", h:1, mode:"ablation", cfg:"standalone GDELT tone Δ", dir:0.0000, sd:0.0000, strict:null, strictSd:null, mae:null, rmse:null }
  ];

  /* ---------------------------------------------------------------- helpers */
  const $ = (sel) => document.querySelector(sel);
  const fmt = (v, d) => (v === null || v === undefined) ? "—" : (typeof v === "number" ? v.toFixed(d === undefined ? 4 : d) : String(v));
  const fmtInt = (v) => v === null ? "—" : Number(v).toLocaleString("en-US");

  const SVGNS = "http://www.w3.org/2000/svg";
  function svgEl(tag, attrs) {
    const el = document.createElementNS(SVGNS, tag);
    Object.entries(attrs || {}).forEach(([k, v]) => el.setAttribute(k, v));
    return el;
  }

  const tooltip = document.createElement("div");
  tooltip.className = "tooltip";
  document.body.appendChild(tooltip);
  function showTip(html, ev) {
    tooltip.innerHTML = html;
    tooltip.classList.add("visible");
    const pad = 16;
    let x = ev.clientX + pad, y = ev.clientY + pad;
    const r = tooltip.getBoundingClientRect();
    if (x + r.width > innerWidth - 8) x = ev.clientX - r.width - pad;
    if (y + r.height > innerHeight - 8) y = ev.clientY - r.height - pad;
    tooltip.style.left = x + "px"; tooltip.style.top = y + "px";
  }
  function hideTip() { tooltip.classList.remove("visible"); }

  /* ---------------------------------------------------------------- generic grouped bar chart */
  function groupedBars(container, groups, series, opts = {}) {
    const W = 820, H = opts.height || 340, padL = 64, padR = 24, padT = 30, padB = 48;
    const plotW = W - padL - padR, plotH = H - padT - padB;
    container.innerHTML = "";
    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
    const maxV = opts.maxV || 1.0;
    const y = (v) => padT + plotH * (1 - v / maxV);

    // gridlines + y labels
    for (let i = 0; i <= 4; i++) {
      const v = (maxV * i) / 4;
      const yy = y(v);
      svg.appendChild(svgEl("line", { x1: padL, y1: yy, x2: W - padR, y2: yy, stroke: "var(--hair)", "stroke-width": 1 }));
      svg.appendChild(svgEl("text", { x: padL - 8, y: yy + 4, "text-anchor": "end", class: "bar-label-t" })).textContent = v.toFixed(2);
    }
    // chance reference line
    if (opts.chance !== undefined) {
      const cy = y(opts.chance);
      svg.appendChild(svgEl("line", { x1: padL, y1: cy, x2: W - padR, y2: cy, stroke: "var(--brass)", "stroke-width": 1.5, "stroke-dasharray": "6 4" }));
      svg.appendChild(svgEl("text", { x: W - padR - 4, y: cy - 6, "text-anchor": "end", class: "bar-label-t" })).textContent = `chance ${opts.chance}`;
    }

    const groupW = plotW / groups.length;
    const nSeries = series.length;
    const barW = Math.min(groupW * 0.72 / nSeries, 64);
    const gap = (groupW - barW * nSeries) / (nSeries + 1);

    groups.forEach((g, gi) => {
      const gx = padL + gi * groupW;
      svg.appendChild(svgEl("text", { x: gx + groupW / 2, y: H - 20, "text-anchor": "middle", class: "bar-label-t" })).textContent = g.label;
      series.forEach((s, si) => {
        const val = g[s.key];
        if (!val) return;
        const v = val[0], sd = val[1] || 0;
        const bx = gx + gap + si * (barW + gap);
        const by = y(Math.min(v, maxV));
        const bh = padT + plotH - by;
        const rect = svgEl("rect", {
          x: bx, y: by, width: barW, height: Math.max(bh, 1),
          fill: s.color, rx: 4
        });
        if (sd > 0) {
          const eTop = y(Math.min(v + sd, maxV));
          const eBottom = y(Math.max(v - sd, 0));
          svg.appendChild(svgEl("line", { x1: bx + barW / 2, y1: eTop, x2: bx + barW / 2, y2: eBottom, stroke: "var(--ink)", "stroke-width": 1.4 }));
          svg.appendChild(svgEl("line", { x1: bx + barW / 2 - 4, y1: eTop, x2: bx + barW / 2 + 4, y2: eTop, stroke: "var(--ink)", "stroke-width": 1.4 }));
          svg.appendChild(svgEl("line", { x1: bx + barW / 2 - 4, y1: eBottom, x2: bx + barW / 2 + 4, y2: eBottom, stroke: "var(--ink)", "stroke-width": 1.4 }));
        }
        svg.appendChild(svgEl("text", { x: bx + barW / 2, y: by - 5, "text-anchor": "middle", class: "bar-val-t" })).textContent = v.toFixed(4);
        rect.addEventListener("mousemove", (ev) => showTip(
          `<strong>${g.label}</strong> · ${s.name}<br>DirAcc <strong>${v.toFixed(4)}</strong> ± ${sd.toFixed(4)}`, ev));
        rect.addEventListener("mouseleave", hideTip);
        svg.appendChild(rect);
      });
    });
    container.appendChild(svg);
  }

  /* ---------------------------------------------------------------- charts */
  let ticker = "aapl";

  function drawHorizon() {
    const d = HORIZON[ticker];
    const groups = [
      { label: "h=1", key: "h1" }, { label: "h=3", key: "h3" }, { label: "h=5", key: "h5" }
    ];
    const series = [
      { key: "overlap", name: "Overlap", color: "var(--ink-2)" },
      { key: "strict", name: "Strict non-overlap", color: "var(--brass)" },
      { key: "momentum", name: "Causal momentum", color: "var(--mut)" }
    ];
    const data = groups.map((g) => ({
      label: g.label,
      overlap: d.overlap[g.key], strict: d.strict[g.key], momentum: d.momentum[g.key]
    }));
    groupedBars($("#chartHorizon"), data, series, { chance: 0.5 });
  }

  function drawEquiv() {
    const e = EQUIV[ticker];
    const groups = [
      { label: "GRU 50/1", key: "gru" },
      { label: "Ridge", key: "ridge" },
      { label: "GBR", key: "gbr" },
      { label: "Momentum", key: "momentum" }
    ];
    const series = [{ key: "v", name: "DirAcc", color: "var(--brass)" }];
    groupedBars($("#chartEquiv"), groups.map(g => ({ label: g.label, v: e[g.key] })), series,
      { chance: 0.5, height: 300 });
  }

  function drawDiag() {
    const d = DIAG[ticker];
    const groups = [
      { label: "mse/log", key: "mse" },
      { label: "strict(0.005)", key: "strict" },
      { label: "hybrid/log", key: "hybrid" },
      { label: "strict hybrid", key: "strictHybrid" },
      { label: "logvol", key: "logvol" }
    ];
    const series = [{ key: "v", name: "DirAcc", color: "var(--ink-2)" }];
    groupedBars($("#chartDiag"), groups.map(g => ({ label: g.label, v: d[g.key] })), series,
      { height: 300 });

    // MAE across horizons
    const m = MAE_HORIZON[ticker];
    const mx = Math.max(m.h1, m.h3, m.h5);
    const W = 400, H = 300, padL = 70, padR = 20, padT = 30, padB = 44;
    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}` });
    const plotW = W - padL - padR, plotH = H - padT - padB;
    const y = (v) => padT + plotH * (1 - v / mx);
    [0, 0.25, 0.5, 0.75, 1].forEach((p) => {
      const v = mx * p, yy = y(v);
      svg.appendChild(svgEl("line", { x1: padL, y1: yy, x2: W - padR, y2: yy, stroke: "var(--hair)", "stroke-width": 1 }));
      svg.appendChild(svgEl("text", { x: padL - 6, y: yy + 4, "text-anchor": "end", class: "bar-label-t" })).textContent = Number(v).toLocaleString();
    });
    ["h1", "h3", "h5"].forEach((h, i) => {
      const bw = 60, gx = padL + (plotW - 3 * 90) / 2 + i * 90;
      const v = m[h], bh = plotH * (v / mx);
      svg.appendChild(svgEl("rect", { x: gx, y: y(v), width: bw, height: bh, fill: "var(--ink-2)", rx: 4 }));
      svg.appendChild(svgEl("text", { x: gx + bw / 2, y: y(v) - 5, "text-anchor": "middle", class: "bar-val-t" })).textContent = Number(v).toLocaleString();
      svg.appendChild(svgEl("text", { x: gx + bw / 2, y: H - 16, "text-anchor": "middle", class: "bar-label-t" })).textContent = h.toUpperCase();
    });
    $("#chartMaeHorizon").innerHTML = "";
    $("#chartMaeHorizon").appendChild(svg);
  }

  function drawAblation() {
    const a = ABLATION[ticker];
    const groups = [
      { label: "Price-only", key: "price" },
      { label: "Price + tech", key: "tech" },
      { label: "+ Lagged GDELT tone", key: "sent" }
    ];
    const series = [{ key: "v", name: "DirAcc", color: "var(--ink-2)" }];
    groupedBars($("#chartAblation"), groups.map(g => ({ label: g.label, v: a[g.key] })), series,
      { chance: 0.5, height: 300 });
    const meta = $("#tickerMeta");
    meta.textContent = `${TICKERS[ticker].label} — ${TICKERS[ticker].long}. ` +
      `Lagged GDELT tone rows: ${a.rows}. Standalone tone Δ = ${fmt(a.toneDelta[0])} ± ${fmt(a.toneDelta[1])}; ` +
      (a.dummyDelta !== null ? `zero-feature dummy control Δ ≈ ${a.dummyDelta.toFixed(3)}.` : "zero-feature dummy control not defined for this ticker.");
  }

  function drawSchematic() {
    const el = $("#schematic");
    el.innerHTML = "";
    const W = 820, H = 300, pad = 20;
    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}` });
    const blocks = [
      { x: 20, w: 200, t: "Raw series", s: "Date · Close · (tone)" },
      { x: 260, w: 200, t: "Chronological split", s: "70 / 15 / 15" },
      { x: 500, w: 200, t: "Scaler", s: "MinMax fitted on TRAIN rows only" },
      { x: 620, w: 190, t: "Windows", s: "lookback = 30 · per split" }
    ];
    blocks.forEach((b, i) => {
      const g = svgEl("g", {});
      g.appendChild(svgEl("rect", { x: b.x, y: 80, width: b.w, height: 90, rx: 12, fill: "var(--card)", stroke: "var(--hair)", "stroke-width": 1.5 }));
      g.appendChild(svgEl("text", { x: b.x + b.w / 2, y: 112, "text-anchor": "middle", class: "bar-label-t" })).textContent = b.t;
      g.appendChild(svgEl("text", { x: b.x + b.w / 2, y: 140, "text-anchor": "middle", class: "bar-val-t", "font-size": "11px" })).textContent = b.s;
      if (i < blocks.length - 1) {
        g.appendChild(svgEl("line", { x1: b.x + b.w + 6, y1: 125, x2: blocks[i + 1].x - 6, y2: 125, stroke: "var(--brass)", "stroke-width": 2 }));
        g.appendChild(svgEl("polygon", { points: `${blocks[i + 1].x - 6},119 ${blocks[i + 1].x - 6},131 ${blocks[i + 1].x + 2},125`, fill: "var(--brass)" }));
      }
      svg.appendChild(g);
    });
    // badges
    const badges = [
      { x: 20, y: 200, t: "maxdiff = 0.0 (timestamp probe)" },
      { x: 260, y: 200, t: "lookback = 30 · boundary-safe" },
      { x: 500, y: 200, t: "train-only transform · zero look-ahead" },
      { x: 620, y: 200, t: "Samsung nRMSE 0.038 vs paper 0.048" }
    ];
    badges.forEach((b) => {
      svg.appendChild(svgEl("text", { x: b.x + 6, y: b.y, class: "bar-val-t", "font-size": "12px" })).textContent = b.t;
    });
    el.appendChild(svg);
  }

  function redraw() { drawHorizon(); drawEquiv(); drawDiag(); drawAblation(); }

  /* ---------------------------------------------------------------- ticker tabs */
  const tabs = document.querySelectorAll(".tab[data-ticker]");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => { t.classList.toggle("is-active", t === tab); t.setAttribute("aria-pressed", t === tab ? "true" : "false"); });
      ticker = tab.dataset.ticker;
      redraw();
    });
  });

  /* ---------------------------------------------------------------- explorer */
  const table = $("#explorerTable");
  const headers = ["Ticker", "Horizon", "Mode", "Config", "DirAcc", "±std", "strict(0.005)", "MAE", "RMSE"];
  function renderExplorer(query = "") {
    const q = query.trim().toLowerCase();
    const head = table.querySelector("thead tr");
    head.innerHTML = "";
    headers.forEach(h => { const th = document.createElement("th"); th.textContent = h; head.appendChild(th); });
    const tbody = table.querySelector("tbody");
    tbody.innerHTML = "";
    ROWS.filter(r => !q || JSON.stringify(r).toLowerCase().includes(q)).forEach(r => {
      const tr = document.createElement("tr");
      const cells = [
        r.t, `h=${r.h}`, r.mode, r.cfg,
        fmt(r.dir), fmt(r.sd), fmt(r.strict, 4), fmtInt(r.mae), fmtInt(r.rmse)
      ];
      cells.forEach(c => { const td = document.createElement("td"); td.textContent = c; tr.appendChild(td); });
      tbody.appendChild(tr);
    });
  }
  $("#explorerQuery").addEventListener("input", (e) => renderExplorer(e.target.value));
  renderExplorer();

  function exportDataset(type) {
    const payload = type === "csv"
      ? [headers.join(",")].concat(ROWS.map(r => headers.map((h, i) => {
          const v = [r.t, `h=${r.h}`, r.mode, r.cfg, r.dir, r.sd, r.strict, r.mae, r.rmse][i];
          return typeof v === "number" ? v : `"${String(v).replace(/"/g, '""')}"`;
        }).join(","))).join("\n")
      : JSON.stringify(ROWS, null, 2);
    const blob = new Blob([payload], { type: type === "csv" ? "text/csv" : "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `verified_metrics.${type}`;
    a.click();
    URL.revokeObjectURL(a.href);
  }
  $("#exportCsv").addEventListener("click", () => exportDataset("csv"));
  $("#exportJson").addEventListener("click", () => exportDataset("json"));

  /* ---------------------------------------------------------------- copy buttons */
  document.querySelectorAll(".copy-btn").forEach(btn => {
    btn.addEventListener("click", async () => {
      const pre = document.getElementById(btn.dataset.target);
      try { await navigator.clipboard.writeText(pre.textContent); btn.textContent = "Copied"; }
      catch { const ta = document.createElement("textarea"); ta.value = pre.textContent; document.body.appendChild(ta); ta.select(); document.execCommand("copy"); ta.remove(); btn.textContent = "Copied"; }
      setTimeout(() => { btn.textContent = "Copy"; }, 1600);
    });
  });

  /* ---------------------------------------------------------------- theme */
  const themeToggle = $("#themeToggle");
  const applyTheme = (t) => {
    document.documentElement.dataset.theme = t;
    localStorage.setItem("gru-theme", t);
    themeToggle.textContent = t === "dark" ? "◑" : "◐";
  };
  applyTheme(localStorage.getItem("gru-theme") ||
    (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));
  themeToggle.addEventListener("click", () =>
    applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));

  /* ---------------------------------------------------------------- reveal */
  const revealables = document.querySelectorAll(".tl-step, .chart-card, .repro-card, .gallery figure");
  if ("IntersectionObserver" in window) {
    const io = new IntersectionObserver((entries) => {
      entries.forEach(e => { if (e.isIntersecting) { e.target.style.opacity = "1"; e.target.style.transform = "none"; io.unobserve(e.target); } });
    }, { threshold: 0.1 });
    revealables.forEach(el => { el.style.opacity = "0"; el.style.transform = "translateY(18px)"; el.style.transition = "opacity .6s var(--ease), transform .6s var(--ease)"; io.observe(el); });
  }

  /* ---------------------------------------------------------------- counters */
  const animateCount = (el) => {
    const target = parseFloat(el.dataset.count);
    const dec = parseInt(el.dataset.decimals || "0", 10);
    const t0 = performance.now(), dur = 1200;
    const step = (now) => {
      const p = Math.min((now - t0) / dur, 1);
      const eased = 1 - Math.pow(1 - p, 3);
      el.textContent = (dec > 0 ? (target * eased).toFixed(dec) : Math.round(target * eased).toString());
      if (p < 1) requestAnimationFrame(step); else el.textContent = dec > 0 ? target.toFixed(dec) : Math.round(target).toString();
    };
    requestAnimationFrame(step);
  };
  document.querySelectorAll(".fact-num").forEach(el => animateCount(el));

  /* ---------------------------------------------------------------- literature corpus synthesis */
  const LIT = {
    facts: [
      [575, "real papers in corpus (all verified)"],
      [18, "seed refs of Padmakumari et al. (2026)"],
      [404, "citing-lineage papers (cite the 18 seeds)"],
      [144, "tight title/abstract-search backfill"],
      [208, "records rejected (dedupe + guards + cull)"],
      [570, "abstracts (99.1%) · 400 open access"],
      [12, "batch notes · every paper accounted"],
      [0, "GDELT-mentioning papers — our ingestion is corpus-novel"]
    ],
    themes: [
      ["T1 · LSTM dominates; GRU is the efficient minority", "LSTM 254 (44.2%) vs GRU 75 (13.0%). GRU efficiency precedents: Thorat 2023 (GRU ~2× faster than BiLSTM), Saud 2020 (GRU best on NEPSE banks, look-back study), Shahi 2020 (normalized LSTM-vs-GRU + significance), Farhadi 2025 (LSTM-GRU + full technical features), Vancsura 2025 (PRISMA: LSTM/GRU/XGBoost/hybrids dominate 4 asset classes)."],
      ["T2 · Sentiment is shallow in time discipline; GDELT absent", "Twitter/social 103, news 101, BERT/FinBERT 53 abstracts; GDELT 0. Memi 2024 (BIST100 tweets, CNN 83.02%), Mehta 2021 (harvesting + DL), Zhang 2023 (DGATS graph + Kronecker fusion), Das 2024 (GNN+sentiment survey). Our 1-day-lagged GDELT tone has no corpus precedent."],
      ["T3 · Price-level targets dominate", "Price level 278 (48.3%) vs returns 82 (14.3%) vs direction 123 (21.4%) vs volatility 17 (3.0%) — our return-target diagnostics (logvol/hybrid, dead-zone DirAcc) align with the minority practice."],
      ["T4 · Multi-day evaluation barely exists", "9 abstracts state multi-day horizons; 0 report non-overlapping evaluation. Friday 2025 (ViT+TFT, time-series CV, 8 indices, h=1/3/7/10) is the closest practice. Our --stride nonoverlap + momentum baseline fills this void."],
      ["T5 · Accuracy-chasing culture", "Classification accuracy 318 (55.3%) vs significance tests 20 (3.5%) vs directional accuracy 12 (2.1%). Raju 2024 (seed) claims 85% accuracy + 175% 1-year return. Honest templates: Shahi 2020, Peng 2021 (124-indicator factor zoo, walk-forward + significance)."],
      ["T6 · Leakage controls named by 2025–26 works", "R-CSAN decoder masking (Sun 2025), rolling validation (Minocha 2026), strict temporal separation + pre-open news (Niharmine 2026), chronological handling (Manhas 2026), temporal-leakage review (Theodorakopoulos 2026) — all converge on our audit's controls."],
      ["T7 · Korea nearly absent", "India 46 · US 43 · China 41 · single ticker 33 · Korea 2 (0.3%). Samsung replication is corpus-rare; dual-market AAPL + Samsung design is justified."],
      ["T8 · Frontier ≠ validity", "Transformer 63 · attention 52 · GNN 35 · xLSTM (Fan 2024) · FNSPID dataset (Dong 2024, KDD, cited 184) — yet Theodorakopoulos 2026: gains depend more on alignment discipline than architectural novelty. Matches our h=1 equivalence (GRU = Ridge = GBR = momentum)."]
    ],
    pitfalls: [
      { id: "P1", pitfall: "Whole-dataset scaling", corpus: "31/575 mention scaling; 0/575 report train-only fitting", paper: "MinMax on whole dataset before split (per audit)", audit: "Train-only scaler; maxdiff = 0.0 timestamp probe" },
      { id: "P2", pitfall: "Overlapping multi-day evaluation", corpus: "9 abstracts mention multi-day; 0 report non-overlapping evaluation", paper: "Not reported — added by our audit", audit: "--stride nonoverlap; momentum baseline; small-n disclosed" },
      { id: "P3", pitfall: "Accuracy inflation / R²-as-accuracy", corpus: "318/575 classification accuracy vs 20/575 significance tests", paper: "MAPE/nRMSE/R² only; no directional or significance metrics", audit: "DirAcc + dead-zone ε + strict(0.005) + chance line + seed std" },
      { id: "P4", pitfall: "No persistence/momentum baseline", corpus: "Persistence in 6 abstracts; momentum rules in 8", paper: "No persistence baseline", audit: "Persistence + causal momentum (AAPL 0.4841; Samsung 0.4837 at h=1)" },
      { id: "P5", pitfall: "Sentiment alignment undisciplined", corpus: "Theodorakopoulos 2026 (temporal leakage); Niharmine 2026 (pre-open news); Minocha 2026 (rolling validation)", paper: "Categorical fusion rule", audit: "1-trading-day causal lag; standalone Δ; dummy control separate" },
      { id: "P6", pitfall: "Unreported hyperparameters", corpus: "Most abstracts silent on config", paper: "GRU hidden/layers unreported", audit: "hidden 50 · layers 1 disclosed; CLI reproducible" },
      { id: "P7", pitfall: "Single-market overfitting", corpus: "Single ticker 33; Korea 2/575 (0.3%)", paper: "Samsung-only", audit: "Dual market AAPL + Samsung, per-market tables" },
      { id: "P8", pitfall: "Volume/volatility inputs underused", corpus: "Volume 12; realized volatility 6", paper: "Close-only features", audit: "Known gap: loader drops OHLCV → experiment E7" }
    ],
    experiments: [
      { id: "E1", t: "Return/directional targets with volatility-aware evaluation", why: "price-level 48.3% vs returns 14.3% vs volatility 3.0%", status: "Partially run: hybrid 0.4957 ± 0.0209; logvol 0.4822 ± 0.0049 (AAPL h=1)" },
      { id: "E2", t: "Non-overlapping multi-day evaluation as standard", why: "0/575 corpus papers report it", status: "Verified: AAPL h=3 0.6494 ± 0.0347 · h=5 0.7333 ± 0.0526 vs momentum 0.5086/0.5000" },
      { id: "E3", t: "Domain sentiment (FNSPID-style) vs GDELT tone ablation", why: "BERT/FinBERT 53 · GDELT 0 · news 101", status: "Pending — FNSPID (Dong 2024, KDD, cited 184) is in corpus" },
      { id: "E4", t: "Significance battery (Diebold–Mariano / Wilcoxon across seeds)", why: "only 20/575 (3.5%) test significance", status: "Pending — template: Shahi 2020, Peng 2021" },
      { id: "E5", t: "Cross-market replication + time-series cross-validation", why: "Korea 0.3% of corpus; Friday 2025's 8-index TSCV", status: "Pending" },
      { id: "E6", t: "Leakage reporting checklist as repo standard", why: "Theodorakopoulos 2026 checklists; R-CSAN masking; Niharmine pre-open rule", status: "Partially: maxdiff = 0.0 probe + train-only scaler already standard" },
      { id: "E7", t: "Volume + realized-volatility inputs", why: "volume 12 · realized vol 6; our loader drops OHLCV", status: "Pending — src/features_research.py" },
      { id: "E8", t: "One attention/TCN baseline on identical splits", why: "attention 52 · TCN 11; R-CSAN ablations", status: "Pending — hypothesis: helps multi-day disjoint, not h=1" }
    ],
    limits: [
      "Abstract-level analysis (full text mostly paywalled) — findings are about reporting practices in abstracts.",
      "5 of 18 seed references are paywalled with no abstract in OpenAlex/Crossref/Semantic Scholar; retained with verified metadata, excluded from abstract-level claims.",
      "Lineage-defined corpus (co-citation with the faculty bibliography) — strong relevance, not an exhaustive PRISMA review.",
      "25 post-audit culls preserved in literature_corpus/_excluded/; 208 total rejections logged in the manifest.",
      "OpenAlex metadata (venues, citations) as of 2026-11-23."
    ],
    papers: [
      { cited: 841, year: 2020, title: "Stock Closing Price Prediction using Machine Learning Techniques", venue: "Procedia Computer Science", doi: "10.1016/j.procs.2020.03.326", seed: true, why: "Faculty paper's most-cited reference: ANN + Random Forest, next-day close, RMSE/MAPE." },
      { cited: 258, year: 2020, title: "Stock Price Forecasting with Deep Learning: A Comparative Study", venue: "Mathematics", doi: "10.3390/math8091441", seed: false, why: "Normalized LSTM-vs-GRU comparison with significance assessment — the honest-comparison template." },
      { cited: 246, year: 2023, title: "TRNN: An efficient time-series recurrent neural network for stock price prediction", venue: "Information Sciences", doi: "10.1016/j.ins.2023.119951", seed: true, why: "Seed reference (paywalled; metadata verified). Time-series RNN variant." },
      { cited: 228, year: 2022, title: "Novel optimization approach for stock price forecasting using multi-layered sequential LSTM", venue: "Applied Soft Computing", doi: "10.1016/j.asoc.2022.109830", seed: true, why: "Seed reference (paywalled; metadata verified). Multi-layered sequential LSTM + optimization." },
      { cited: 226, year: 2021, title: "Harvesting social media sentiment analysis to enhance stock market prediction using deep learning", venue: "PeerJ Computer Science", doi: "10.7717/peerj-cs.476", seed: false, why: "Social-media sentiment harvesting pipeline feeding deep learning." },
      { cited: 184, year: 2024, title: "FNSPID: A Comprehensive Financial News Dataset in Time Series", venue: "ACM SIGKDD (KDD)", doi: "10.1145/3637528.3671629", seed: false, why: "Labelled financial-news dataset — the E3 domain-sentiment candidate source." },
      { cited: 137, year: 2022, title: "Stock market prediction based on statistical data using machine learning algorithms", venue: "J. King Saud University - Science", doi: "10.1016/j.jksus.2022.101940", seed: true, why: "Seed: RF + SVM; 80.3% accuracy claim — accuracy-culture example." },
      { cited: 135, year: 2020, title: "Analysis of look back period for stock price prediction with RNN variants", venue: "Procedia Computer Science", doi: "10.1016/j.procs.2020.03.419", seed: true, why: "Seed: Vanilla RNN vs LSTM vs GRU on NEPSE banks; GRU most successful; look-back analysis." },
      { cited: 124, year: 2021, title: "Feature selection and deep neural networks for stock price direction forecasting using technical analysis indicators", venue: "Machine Learning with Applications", doi: "10.1016/j.mlwa.2021.100060", seed: false, why: "124-indicator factor zoo + feature selection; walk-forward, baseline, significance." },
      { cited: 122, year: 2025, title: "Deep learning for financial forecasting: A review of recent trends", venue: "International Review of Economics & Finance", doi: "10.1016/j.iref.2025.104719", seed: false, why: "Review of 187 Scopus studies (2020–2024): MLP, RNN variants, CNN, transformers." },
      { cited: 113, year: 2024, title: "Deep-learning-based stock market prediction incorporating ESG sentiment and technical indicators", venue: "Scientific Reports", doi: "10.1038/s41598-024-61106-2", seed: false, why: "ESG sentiment fused with technical indicators." },
      { cited: 109, year: 2024, title: "A Novel Variant of LSTM Stock Prediction Method Incorporating Attention Mechanism", venue: "Mathematics", doi: "10.3390/math12070945", seed: true, why: "Seed: AMV-LSTM — attention-coupled LSTM variant, Adam, generalization claims." },
      { cited: 101, year: 2024, title: "Stock price prediction: comparison of different moving average techniques using deep learning model", venue: "Neural Computing and Applications", doi: "10.1007/s00521-023-09369-0", seed: true, why: "Seed (paywalled; metadata verified): moving-average technique comparison with DL." },
      { cited: 98, year: 2024, title: "Integrating sentiment analysis with graph neural networks for enhanced stock prediction: A comprehensive survey", venue: "Decision Analytics Journal", doi: "10.1016/j.dajour.2024.100417", seed: true, why: "Seed: GNN + sentiment survey; names data collection/preprocessing/annotation as open challenges." },
      { cited: 91, year: 2024, title: "How effective is machine learning in stock market predictions?", venue: "Heliyon", doi: "10.1016/j.heliyon.2024.e24123", seed: false, why: "ML comparison across NYSE100, NIKKEI225, FTSE100, CAC40, DAX30, FTSE MIB, TSX." },
      { cited: 85, year: 2022, title: "A Novel Ensemble Learning Approach for Stock Market Prediction Based on Sentiment Analysis and the Sliding Window Method", venue: "IEEE Trans. Computational Social Systems", doi: "10.1109/tcss.2022.3182375", seed: false, why: "Ensemble + sliding window + financial news." },
      { cited: 84, year: 2024, title: "A deep fusion model for stock market prediction with news headlines and time series data", venue: "Neural Computing and Applications", doi: "10.1007/s00521-024-10303-1", seed: false, why: "Deep fusion of news headlines + time series." },
      { cited: 80, year: 2024, title: "Advanced Stock Price Prediction with xLSTM-Based Models: Improving Long-Term Forecasting", venue: "ISCMI", doi: "10.1109/iscmi63661.2024.10851608", seed: false, why: "xLSTM — modern RNN revival for long-term stock forecasting." },
      { cited: 79, year: 2025, title: "A Hybrid LSTM-GRU Model for Stock Price Prediction", venue: "IEEE Access", doi: "10.1109/access.2025.3586558", seed: false, why: "Hybrid LSTM-GRU leveraging the full range of technical analysis features." },
      { cited: 79, year: 2024, title: "A Systematic Review on Graph Neural Network-based Methods for Stock Market Forecasting", venue: "ACM Computing Surveys", doi: "10.1145/3696411", seed: false, why: "GNN survey; inter-stock dependence beyond single-ticker models." },
      { cited: 74, year: 2025, title: "Navigating AI-Driven Financial Forecasting: A Systematic Review of Current Status and Critical Research Gaps", venue: "Forecasting", doi: "10.3390/forecast7030036", seed: false, why: "PRISMA SLR across equities, crypto, commodities, FX; LSTM/GRU/XGBoost/hybrids dominate." },
      { cited: 49, year: 2024, title: "Comparative Study for Sentiment Analysis of Financial Tweets with Deep Learning Methods", venue: "Applied Sciences", doi: "10.3390/app14020588", seed: true, why: "Seed: BIST100 tweets; CNN best (83.02% binary); GRU-CNN competitive." },
      { cited: 44, year: 2023, title: "A Dynamic Attributes-driven Graph Attention Network Modeling on Behavioral Finance for Stock Prediction", venue: "ACM TKDD", doi: "10.1145/3611311", seed: true, why: "Seed: DGATS — dynamic graph attention + Kronecker tensor fusion of sentiment, transactions, text." },
      { cited: 32, year: 2025, title: "RCSAN residual enhanced channel spatial attention network for stock price forecasting", venue: "Scientific Reports", doi: "10.1038/s41598-025-06885-y", seed: false, why: "Decoder masking prevents future information leakage; full ablation table; cross-market." },
      { cited: 26, year: 2023, title: "Study on the sentimental influence on Indian stock price", venue: "Heliyon", doi: "10.1016/j.heliyon.2023.e22788", seed: true, why: "Seed: modified VADER + Shannon/Rényi entropy causality between news sentiment and price." },
      { cited: 17, year: 2025, title: "A Multi-Modal Approach Using a Hybrid Vision Transformer and Temporal Fusion Transformer Model for Stock Price Movement Classification", venue: "IEEE Access", doi: "10.1109/access.2025.3589063", seed: false, why: "ViT + TFT; time-series cross-validation; 8 indices; horizons 1/3/7/10 days." },
      { cited: 17, year: 2024, title: "A comprehensive review on sentiment analysis of social/web media big data for stock market prediction", venue: "Int. J. Systems Assurance Engineering and Management", doi: "10.1007/s13198-023-02214-6", seed: true, why: "Seed review (abstract backfilled from Springer): social/web media sentiment + big data analytics." },
      { cited: 1, year: 2026, title: "Big Data and Graph Deep Learning for Financial Decision Support from Social Networks: A Critical Review", venue: "Electronics", doi: "10.3390/electronics15071405", seed: false, why: "Names temporal leakage, attention confounds, visibility effects; demands auditable pipelines + checklists." },
      { cited: 0, year: 2026, title: "Deep Learning for Morocco Stock Market Prediction Based on News Sentiment Analysis", venue: "IRASET", doi: "10.1109/iraset68627.2026.11538843", seed: false, why: "MASI index; strict temporal separation; news only before market open; CNN-LSTM 78.6% direction; SHAP." },
      { cited: 0, year: 2026, title: "AI-Powered Sentiment Analysis for Stock Market Prediction in Renewable Energy", venue: "Int. J. Software & Hardware Research in Engineering", doi: "10.26821/ijshre.14.09.2026.140904", seed: false, why: "FinBERT + decay weights + rolling validation eliminating look-ahead bias; h=1–5." },
      { cited: 0, year: 2026, title: "An Explainable Hybrid Artificial Intelligence Framework for Stock Market Prediction Using Financial Time-Series and Market Sentiment", venue: "Zenodo (dissertation)", doi: "10.5281/zenodo.22065590", seed: false, why: "Chronological data handling to prevent information leakage; XAI emphasis." },
      { cited: 27, year: 2023, title: "Stock Price Prediction Using GRU and BiLSTM Models", venue: "LNNS 641 (BITS-EEE-CON 2022)", doi: "10.1007/978-981-99-0483-9_9", seed: false, why: "Related work: BiLSTM slightly better, but GRU ~2× faster — the efficiency precedent for GRU." }
    ]
  };

  function renderLiterature() {
    const facts = $("#litFacts");
    LIT.facts.forEach(([n, label]) => {
      const d = document.createElement("div");
      d.className = "fact lit-fact";
      const num = document.createElement("span");
      num.className = "fact-num";
      num.textContent = typeof n === "number" ? n.toLocaleString("en-US") : String(n);
      const lab = document.createElement("span");
      lab.className = "fact-label";
      lab.textContent = label;
      d.append(num, lab);
      facts.appendChild(d);
    });
    const themes = $("#litThemes");
    themes.innerHTML = "<h3>Cross-corpus themes (grounded in abstracts)</h3>";
    LIT.themes.forEach(([t, dsc]) => {
      const el = document.createElement("div");
      el.className = "lit-item";
      const h = document.createElement("h4");
      h.textContent = t;
      const p = document.createElement("p");
      p.textContent = dsc;
      el.append(h, p);
      themes.appendChild(el);
    });
    const pit = $("#litPitfalls");
    pit.innerHTML = "<h3>Methodological pitfalls → faculty paper → our audit</h3>";
    const table = document.createElement("table");
    const thead = document.createElement("thead");
    thead.innerHTML = "<tr><th>Pitfall</th><th>Corpus evidence</th><th>Faculty paper (2026)</th><th>Our leak-free audit</th></tr>";
    const tbody = document.createElement("tbody");
    LIT.pitfalls.forEach(r => {
      const tr = document.createElement("tr");
      [r.id + " · " + r.pitfall, r.corpus, r.paper, r.audit].forEach(c => {
        const td = document.createElement("td");
        td.textContent = c;
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    table.append(thead, tbody);
    pit.appendChild(table);
    const exp = $("#litExperiments");
    exp.innerHTML = "<h3>Prioritized next experiments</h3><ul></ul>";
    const ul = exp.querySelector("ul");
    LIT.experiments.forEach(e => {
      const li = document.createElement("li");
      const b = document.createElement("b");
      b.textContent = e.id + " — " + e.t + ". ";
      const i = document.createElement("i");
      i.textContent = "Corpus: " + e.why + ". ";
      const s = document.createElement("span");
      s.textContent = e.status;
      li.append(b, i, s);
      ul.appendChild(li);
    });
    const lim = $("#litLimits");
    lim.innerHTML = "<h3>Limits &amp; integrity notes</h3><ul></ul>";
    const limUl = lim.querySelector("ul");
    LIT.limits.forEach(l => {
      const li = document.createElement("li");
      li.textContent = l;
      limUl.appendChild(li);
    });
    const wrap = $("#litPapers");
    LIT.papers.forEach(p => {
      const d = document.createElement("div");
      d.className = "lit-paper";
      const t = document.createElement("div");
      t.className = "lp-title";
      t.textContent = p.title;
      if (p.seed) {
        const badge = document.createElement("span");
        badge.className = "lp-seed";
        badge.textContent = "FACULTY BIBLIOGRAPHY";
        t.appendChild(badge);
      }
      const meta = document.createElement("div");
      meta.className = "lp-meta";
      meta.textContent = p.year + " · " + p.venue + " · cited " + p.cited + " · doi " + p.doi;
      const why = document.createElement("div");
      why.className = "lp-why";
      why.textContent = p.why;
      d.append(t, meta, why);
      wrap.appendChild(d);
    });
    $("#litFoot").textContent = "All counts from literature_corpus/_manifest.json + _corpus_stats.json (regenerated by scripts/). 5 paywalled seed abstracts disclosed in the synthesis. This section is covered by the global search bar above.";
  }

  /* ---------------------------------------------------------------- paper vs audit matrix */
  const MATRIX = [
    { label: "Samsung nRMSE (test)", paper: "0.048 (reported)", val: 0.038,
      audit: "0.038 (verified)", note: "Same asset class; smaller is better",
      tick: "005930.KS", h: "", mode: "" },
    { label: "DirAcc h=1 · AAPL", paper: "not reported", val: 0.4832,
      audit: "GRU 0.4832 ± 0.0089 (= Ridge; momentum 0.4841)",
      note: "At the daily noise floor", tick: "AAPL", h: "h=1", mode: "" },
    { label: "DirAcc h=1 · Samsung", paper: "not reported", val: 0.4224,
      audit: "GRU 0.4224 ± 0.0014 (= Ridge; momentum 0.4837)",
      note: "Below causal momentum", tick: "005930.KS", h: "h=1", mode: "" },
    { label: "DirAcc h=3 · AAPL", paper: "—", val: 0.6494,
      audit: "Overlap 0.7762 ± 0.0118 · Non-overlap 0.6494 ± 0.0347 · Momentum 0.5086",
      note: "Overlapping windows inflate; disjoint samples small (~tens of points)",
      tick: "AAPL", h: "h=3", mode: "Non-Overlap" },
    { label: "DirAcc h=5 · AAPL", paper: "—", val: 0.7333,
      audit: "Overlap 0.7839 ± 0.0265 · Non-overlap 0.7333 ± 0.0526 · Momentum 0.5000",
      note: "Same multi-day caveat", tick: "AAPL", h: "h=5", mode: "Non-Overlap" },
    { label: "DirAcc h=3 · Samsung", paper: "—", val: 0.6254,
      audit: "Overlap 0.6538 ± 0.0061 · Non-overlap 0.6254 ± 0.0083 · Momentum 0.5044",
      note: "Disjoint windows, small sample", tick: "005930.KS", h: "h=3", mode: "Non-Overlap" },
    { label: "DirAcc h=5 · Samsung", paper: "—", val: 0.6422,
      audit: "Overlap 0.6785 ± 0.0028 · Non-overlap 0.6422 ± 0.0302 · Momentum 0.5000",
      note: "Same multi-day caveat", tick: "005930.KS", h: "h=5", mode: "Non-Overlap" },
    { label: "Sentiment ablation h=1 · AAPL", paper: "categorical fusion rule (paper)", val: 0.5067,
      audit: "price 0.4832 ± 0.0089 · +tech 0.468 ± 0.011 · +GDELT tone 0.5067 ± 0.0303 · standalone Δ −0.0022 ± 0.0154",
      note: "Within seed noise (n=3)", tick: "AAPL", h: "h=1", mode: "" },
    { label: "Sentiment ablation h=1 · Samsung", paper: "categorical fusion rule (paper)", val: 0.492,
      audit: "price 0.4224 ± 0.0014 · +tech 0.448 ± 0.010 · +GDELT tone 0.492 ± 0.021 · standalone Δ 0.0000 ± 0.0000",
      note: "Within seed noise (n=3)", tick: "005930.KS", h: "h=1", mode: "" },
    { label: "Zero-feature dummy control", paper: "n/a", val: 0.003,
      audit: "Δ ≈ +0.003 (AAPL) — sentiment-ablation-stock-only",
      note: "Dummy control, NOT a true sentiment ablation", tick: "AAPL", h: "h=1", mode: "" },
    { label: "Split / scaler / windows", paper: "chronological 70/15/15; whole-data MinMax; step 30",
      val: null, audit: "train-only scaler; per-split windows; maxdiff = 0.0",
      note: "Leak-free engineering improvement", tick: "", h: "", mode: "" },
    { label: "Architecture disclosure", paper: "hyperparameters unreported", val: null,
      audit: "GRU hidden 50 · layers 1 disclosed", note: "Closes reproducibility gap", tick: "", h: "", mode: "" }
  ];

  const matrixState = { ticker: "all", h: "all", mode: "all", q: "", sortKey: "val", sortDir: -1 };

  function matrixVisible(r) {
    if (matrixState.ticker !== "all" && r.tick !== matrixState.ticker) return false;
    if (matrixState.h !== "all" && r.h !== matrixState.h) return false;
    if (matrixState.mode !== "all" && r.mode && r.mode !== matrixState.mode) return false;
    if (matrixState.q) {
      const hay = JSON.stringify(r).toLowerCase();
      if (!hay.includes(matrixState.q)) return false;
    }
    return true;
  }

  function renderMatrix() {
    const tbody = document.querySelector("#auditTable tbody");
    const foot = $("#matrixFoot");
    const empty = $("#matrixEmpty");
    let rows = MATRIX.filter(matrixVisible);
    const key = matrixState.sortKey;
    rows.sort((a, b) => {
      const av = key === "label" ? a.label : a.val;
      const bv = key === "label" ? b.label : b.val;
      if (av === null) return 1; if (bv === null) return -1;
      if (typeof av === "number" && typeof bv === "number") return (av - bv) * matrixState.sortDir;
      return String(av).localeCompare(String(bv)) * matrixState.sortDir;
    });
    tbody.innerHTML = "";
    rows.forEach(r => {
      const tr = document.createElement("tr");
      [
        r.label, r.paper, r.audit, r.note
      ].forEach(c => { const td = document.createElement("td"); td.textContent = c; tr.appendChild(td); });
      tbody.appendChild(tr);
    });
    foot.textContent = "Above = verified numbers from seeds 42/43/44 (48/48 runs). Empty paper cells mean the 2026 paper did not report that metric.";
    empty.hidden = rows.length > 0;
    // sort arrows
    document.querySelectorAll("#auditTable th.sortable").forEach(th => {
      th.classList.toggle("asc", th.dataset.key === matrixState.sortKey && matrixState.sortDir === 1);
      th.classList.toggle("desc", th.dataset.key === matrixState.sortKey && matrixState.sortDir === -1);
    });
  }

  document.querySelectorAll("#paper-audit .pill").forEach(pill => {
    pill.addEventListener("click", () => {
      const group = pill.dataset.f;
      document.querySelectorAll(`#paper-audit .pill[data-f="${group}"]`).forEach(p => p.classList.remove("is-active"));
      pill.classList.add("is-active");
      matrixState[group === "ticker" ? "ticker" : group] = pill.dataset.v;
      renderMatrix();
    });
  });
  document.querySelectorAll("#auditTable th.sortable").forEach(th => {
    th.addEventListener("click", () => {
      const k = th.dataset.key;
      if (matrixState.sortKey === k) matrixState.sortDir *= -1;
      else { matrixState.sortKey = k; matrixState.sortDir = -1; }
      renderMatrix();
    });
  });

  /* ---------------------------------------------------------------- global live search */
  const searchInput = $("#search-metrics");
  const searchCount = $("#searchCount");
  const SEARCHABLES = ".chart-card, .tl-step, .repro-card, .lit-card, .lit-paper, .lit-fact";

  function applySearch() {
    const q = searchInput.value.trim().toLowerCase();
    matrixState.q = q;
    renderMatrix();
    let visibleCards = 0, totalCards = 0;
    document.querySelectorAll(SEARCHABLES).forEach(el => {
      totalCards++;
      const hit = !q || el.textContent.toLowerCase().includes(q);
      el.classList.toggle("is-hidden", !hit);
      if (hit) visibleCards++;
    });
    searchCount.textContent = `Showing ${matrixVisibleRows()} of ${MATRIX.length} matrix rows · ${visibleCards}/${totalCards} cards & steps`;
  }
  function matrixVisibleRows() { return MATRIX.filter(matrixVisible).length; }
  searchInput.addEventListener("input", applySearch);
  $("#searchClear").addEventListener("click", () => { searchInput.value = ""; applySearch(); searchInput.focus(); });
  document.addEventListener("keydown", (e) => {
    const typing = document.activeElement.tagName === "INPUT" || document.activeElement.tagName === "TEXTAREA";
    if (e.key === "/" && !typing) {
      e.preventDefault(); searchInput.focus();
    }
    if (e.key === "Escape" && typing) {
      searchInput.blur();
    }
  });
  $("#matrixReset").addEventListener("click", () => {
    document.querySelectorAll("#paper-audit .pill").forEach(p => {
      const active = p.dataset.v === "all";
      p.classList.toggle("is-active", active);
    });
    matrixState.ticker = "all"; matrixState.h = "all"; matrixState.mode = "all";
    searchInput.value = ""; applySearch();
  });

  renderMatrix();
  renderLiterature();
  redraw();
})();
