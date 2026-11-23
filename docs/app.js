/* Improving GRU Sentiment Stock Forecasting — showcase scripts (no dependencies). */
(() => {
  "use strict";

  /* Verified data — seeds 42/43/44, commit 55cadea */
  const DATA = {
    aapl: {
      overlap:   { h1: 0.4832, h3: 0.7762, h5: 0.7839 },
      strict:    { h1: 0.4832, h3: 0.6494, h5: 0.7333 },
      momentum:  { h1: 0.4841, h3: 0.5086, h5: 0.5000 }
    },
    samsung: {
      overlap:   { h1: 0.4224, h3: 0.6538, h5: 0.6785 },
      strict:    { h1: 0.4224, h3: 0.6254, h5: 0.6422 },
      momentum:  { h1: 0.4837, h3: 0.5044, h5: 0.5000 }
    }
  };
  const STD = {
    aapl:   { overlap: { h1: "± 0.0089", h3: "± 0.0118", h5: "± 0.0265" },
              strict:  { h1: "± 0.0089", h3: "± 0.0347", h5: "± 0.0526" },
              momentum: { h1: "± 0.0000", h3: "± 0.0000", h5: "± 0.0000" } },
    samsung: { overlap: { h1: "± 0.0014", h3: "± 0.0061", h5: "± 0.0028" },
              strict:  { h1: "± 0.0014", h3: "± 0.0083", h5: "± 0.0302" },
              momentum: { h1: "± 0.0000", h3: "± 0.0000", h5: "± 0.0000" } }
  };
  const CAVEAT = {
    overlap:  "Sliding windows — targets overlap across steps, so these numbers are inflated and not independent predictions.",
    strict:   "Strict non-overlapping (stride = horizon). Disjoint test windows are small (~tens of points); the remaining lift reflects multi-day trend persistence, not claimed tradeable alpha.",
    momentum:  "Causal baseline: sign of the previous h-return (available before the close, no look-ahead)."
  };

  let ticker = "aapl";
  let mode = "overlap";

  /* ------------------------------------------------ theme toggle */
  const themeToggle = document.getElementById("themeToggle");
  const applyTheme = (t) => {
    document.documentElement.dataset.theme = t;
    localStorage.setItem("gru-theme", t);
    themeToggle.textContent = t === "dark" ? "◑" : "◐";
  };
  const saved = localStorage.getItem("gru-theme");
  applyTheme(saved || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));
  themeToggle.addEventListener("click", () =>
    applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark")
  );

  /* ------------------------------------------------ reveal observer */
  const revealables = document.querySelectorAll(".card-reveal, .reveal-target");
  if ("IntersectionObserver" in window) {
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) {
          e.target.classList.add("revealed");
          io.unobserve(e.target);
        }
      });
    }, { threshold: 0.15 });
    revealables.forEach((el) => io.observe(el));
  } else {
    revealables.forEach((el) => el.classList.add("revealed"));
  }

  /* ------------------------------------------------ animated counters */
  const fmt = (v, d) => d > 0 ? v.toFixed(d) : Math.round(v).toString();
  const animateCount = (el) => {
    const target = parseFloat(el.dataset.count);
    const dec = parseInt(el.dataset.decimals || "0", 10);
    const dur = 1200; const t0 = performance.now();
    const step = (now) => {
      const p = Math.min((now - t0) / dur, 1);
      const eased = 1 - Math.pow(1 - p, 3);
      el.textContent = fmt(target * eased, dec);
      if (p < 1) requestAnimationFrame(step);
      else el.textContent = fmt(target, dec);
    };
    requestAnimationFrame(step);
  };
  const counters = document.querySelectorAll(".fact-num");
  const cio = "IntersectionObserver" in window
    ? new IntersectionObserver((entries) => {
        entries.forEach((e) => { if (e.isIntersecting) { animateCount(e.target); cio.unobserve(e.target); } });
      }, { threshold: 0.5 })
    : null;
  counters.forEach((el) => (cio ? cio.observe(el) : animateCount(el)));

  /* ------------------------------------------------ bars */
  const bars = document.getElementById("bars");
  const caveat = document.getElementById("chartCaveat");

  const renderBars = () => {
    const values = DATA[ticker][mode];
    const stdev = STD[ticker][mode];
    bars.innerHTML = "";
    ["h1", "h3", "h5"].forEach((h, i) => {
      const v = values[h];
      const row = document.createElement("div");
      row.className = "bar-row";
      const label = document.createElement("span");
      label.className = "bar-label";
      label.textContent = h.toUpperCase();
      const track = document.createElement("div");
      track.className = "bar-track";
      const fill = document.createElement("div");
      fill.className = "bar-fill";
      fill.style.width = "0%";
      track.appendChild(fill);
      const val = document.createElement("span");
      val.className = "bar-value";
      val.textContent = v.toFixed(4) + " " + stdev[h];
      row.append(label, track, val);
      bars.appendChild(row);
      requestAnimationFrame(() => {
        requestAnimationFrame(() => { fill.style.width = (v * 100) + "%"; });
      });
    });
    caveat.textContent = CAVEAT[mode];
  };

  /* ------------------------------------------------ tabs */
  const tabs = document.querySelectorAll(".tab");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const group = tab.parentElement.classList.contains("tabs-sub") ? "mode" : "ticker";
      const value = tab.dataset.ticker || tab.dataset.mode;
      tab.parentElement.querySelectorAll(".tab").forEach((t) => {
        t.classList.remove("is-active");
        t.setAttribute("aria-selected", "false");
      });
      tab.classList.add("is-active");
      tab.setAttribute("aria-selected", "true");
      if (group === "ticker") ticker = value; else mode = value;
      renderBars();
    });
  });

  renderBars();
})();
