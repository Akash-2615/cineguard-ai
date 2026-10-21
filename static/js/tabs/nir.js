import { api, el, field } from "../api.js";
import { drawSeries } from "../charts.js";

let timer = 0;
let paused = false;
let track = "";
let modeBefore = "simulated";
const scores = [];
const counts = { accepted: [], rejected: [] };

export function mount(root) {
  root.className = "grid";
  const ribbon = el("p", "ribbon", "Simulated infrared. Not a real camera.");
  ribbon.id = "nir-ribbon";
  const views = el("div", "views");
  const figures = {};
  for (const [view, caption] of [["original", "Original"], ["nir", "Simulated NIR"], ["mask", "Glint mask"], ["heat", "Heatmap"]]) {
    const figure = el("figure");
    const img = el("img");
    img.alt = caption;
    figure.append(img, el("figcaption", "", caption));
    figures[view] = img;
    views.append(figure);
  }
  const controls = el("div", "controls");
  const trackSelect = el("select");
  trackSelect.append(new Option("Full frame", ""));
  trackSelect.addEventListener("change", () => { track = trackSelect.value; });
  const pause = el("button", "", "Pause refresh");
  pause.type = "button";
  pause.addEventListener("click", () => {
    paused = !paused;
    pause.textContent = paused ? "Resume refresh" : "Pause refresh";
    pause.setAttribute("aria-pressed", paused ? "true" : "false");
  });
  controls.append(field("Track crop", trackSelect), pause, sliderGroup(), detectorGroup());
  const table = el("table");
  table.id = "nir-candidates";
  table.append(el("caption", "", "Glint candidates"));
  const chart = el("canvas");
  chart.id = "nir-chart";
  const notes = el("details");
  notes.append(el("summary", "", "How this view works"), explainer());
  root.append(ribbon, views, controls, notes, table, chart);
  root._figures = figures;
  root._tracks = trackSelect;
  timer = window.setInterval(() => refresh(root), 1000);
}

function sliderGroup() {
  const box = el("div", "controls");
  box.append(el("h3", "", "Illuminator"));
  const specs = [
    ["intensity", "Intensity", 0, 1, 0.05],
    ["vignette", "Vignette", 0, 1, 0.05],
    ["gain", "Gain", 0.1, 4, 0.1],
    ["synthetic_radius", "Synthetic radius", 1, 30, 0.5],
    ["synthetic_intensity", "Synthetic intensity", 0, 1, 0.05],
  ];
  for (const [key, label, min, max, step] of specs) box.append(nirSlider(key, label, min, max, step));
  const wave = el("select");
  wave.append(new Option("850 nm label", "850"), new Option("940 nm label", "940"));
  wave.addEventListener("change", () => api("/api/v1/nir/params", { method: "POST", json: { wavelength_nm: Number(wave.value) } }));
  const synthetic = el("input");
  synthetic.type = "checkbox";
  synthetic.addEventListener("change", () => api("/api/v1/nir/params", { method: "POST", json: { synthetic_glint: synthetic.checked } }));
  box.append(field("Wavelength label", wave), field("Synthetic glint (demo only)", synthetic));
  box.append(el("p", "", "Synthetic glints are injected for demo only. They are not a lens."));
  return box;
}

function detectorGroup() {
  const box = el("div", "controls");
  box.append(el("h3", "", "Detector"));
  const specs = [
    ["percentile", "Percentile", 90, 99.9, 0.1],
    ["min_area", "Min area", 1, 80, 1],
    ["max_area", "Max area", 4, 400, 1],
    ["circularity_min", "Circularity", 0, 1, 0.05],
    ["contrast_min", "Contrast", 0, 1, 0.05],
    ["persistence_m", "Persistence M", 1, 10, 1],
    ["persistence_n", "Persistence N", 1, 15, 1],
    ["clahe_clip", "CLAHE clip", 0.5, 8, 0.1],
  ];
  for (const [key, label, min, max, step] of specs) box.append(nirSlider(key, label, min, max, step));
  const clahe = el("input");
  clahe.type = "checkbox";
  clahe.checked = true;
  clahe.addEventListener("change", () => api("/api/v1/nir/params", { method: "POST", json: { clahe: clahe.checked } }));
  box.append(field("CLAHE", clahe));
  for (const name of ["conservative", "balanced", "sensitive"]) {
    const button = el("button", "", name[0].toUpperCase() + name.slice(1));
    button.type = "button";
    button.addEventListener("click", () => api("/api/v1/nir/params", { method: "POST", json: { preset: name } }));
    box.append(button);
  }
  const compare = el("button", "", "Compare with NIR off");
  compare.type = "button";
  compare.addEventListener("click", async () => {
    const config = await api("/api/v1/config");
    if (config.nir_mode === "off") {
      await api("/api/v1/config", { method: "POST", json: { nir_mode: modeBefore } });
      compare.textContent = "Compare with NIR off";
    } else {
      modeBefore = config.nir_mode;
      await api("/api/v1/config", { method: "POST", json: { nir_mode: "off" } });
      compare.textContent = "Restore NIR";
    }
  });
  box.append(compare);
  return box;
}

function nirSlider(key, label, min, max, step) {
  const input = el("input");
  input.type = "range"; input.min = String(min); input.max = String(max); input.step = String(step); input.value = String(min);
  input.addEventListener("change", () => {
    const value = key.startsWith("persistence") ? parseInt(input.value, 10) : Number(input.value);
    api("/api/v1/nir/params", { method: "POST", json: { [key]: value } });
  });
  return field(label, input);
}

function explainer() {
  const card = el("div", "panel");
  card.append(el("h3", "", "Why a retroreflector shows up"));
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 360 80");
  svg.setAttribute("width", "100%");
  svg.setAttribute("height", "80");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "IR illuminator, camera lens, and retroreflected spot");
  const parts = [
    ["20", "30", "IR illuminator"],
    ["150", "30", "Camera lens"],
    ["280", "30", "Retroreflected spot"],
  ];
  for (const [x, y, label] of parts) {
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("cx", x); circle.setAttribute("cy", "24"); circle.setAttribute("r", "10");
    circle.setAttribute("fill", "#f59e0b");
    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x", x); text.setAttribute("y", "58"); text.setAttribute("fill", "currentColor");
    text.setAttribute("font-size", "11"); text.setAttribute("text-anchor", "middle");
    text.textContent = label;
    svg.append(circle, text);
  }
  card.append(svg);
  card.append(el("p", "", "An illuminator beside the lens lights the audience. A phone lens sends a tight beam back into the camera. A screen emits little near-infrared, so the same spot is hard to fake with a lit display. In the dark, plain CCTV mostly sees a silhouette."));
  card.append(el("h3", "", "Real hardware checklist"));
  const list = el("ul");
  for (const item of ["850 nm or 940 nm illuminator", "IR-pass filter on the lens", "Global shutter", "Exposure lock"]) {
    list.append(el("li", "", item));
  }
  card.append(list);
  return card;
}

async function refresh(root) {
  if (paused || root.hidden) return;
  const stamp = Date.now();
  const query = track ? `&track=${encodeURIComponent(track)}` : "";
  const original = root._figures.original;
  if (original) showStill(original, `/api/v1/frame.jpg?t=${stamp}`);
  for (const view of ["nir", "mask", "heat"]) {
    const img = root._figures[view];
    if (img) showStill(img, `/api/v1/nir/frame.jpg?view=${view}${query}&t=${stamp}`);
  }
  try {
    const payload = await api("/api/v1/nir/candidates");
    renderCandidates(payload.candidates || []);
    const accepted = (payload.candidates || []).filter((item) => item.accepted).length;
    const rejected = (payload.candidates || []).length - accepted;
    counts.accepted.push(accepted);
    counts.rejected.push(rejected);
    if (counts.accepted.length > 30) {
      counts.accepted.shift();
      counts.rejected.shift();
    }
    const best = Math.max(0, ...(payload.candidates || []).map((item) => item.score || 0));
    scores.push(best);
    if (scores.length > 30) scores.shift();
    const chart = document.getElementById("nir-chart");
    if (chart) drawSeries(chart, [{ values: scores }, { values: counts.accepted.map((n) => n / 5) }, { values: counts.rejected.map((n) => n / 5) }]);
  } catch (error) {
    const table = document.getElementById("nir-candidates");
    if (table) table.replaceChildren(el("caption", "error", error.message));
  }
}

function showStill(img, url) {
  if (img.dataset.next === url) return;
  img.dataset.next = url;
  const incoming = new Image();
  incoming.onload = () => {
    if (img.dataset.next === url) img.src = url;
  };
  incoming.src = url;
}

let candidateKey = "";

function renderCandidates(items) {
  const table = document.getElementById("nir-candidates");
  if (!table) return;
  const key = items.length
    ? items.slice(0, 200).map((item) => `${item.index}:${item.accepted}:${item.score}:${item.rejection_reason || ""}`).join("|")
    : "empty";
  if (key === candidateKey) return;
  candidateKey = key;
  table.replaceChildren();
  const head = el("tr");
  for (const name of ["Thumb", "Area", "Peak", "Circularity", "Contrast", "Score", "Synthetic", "Decision"]) head.append(el("th", "", name));
  table.append(head);
  if (!items.length) {
    table.append(el("caption", "empty", "No candidates on this frame."));
    return;
  }
  for (const item of items.slice(0, 200)) {
    const row = el("tr");
    const thumb = el("td");
    const img = el("img");
    img.alt = "candidate";
    img.src = `/api/v1/nir/candidates/${item.index}.jpg`;
    img.width = 48;
    thumb.append(img);
    row.append(thumb);
    for (const value of [item.area, item.peak, item.circularity, item.contrast, item.score, item.synthetic ? "synthetic" : "observed"]) {
      row.append(el("td", "", String(value)));
    }
    row.append(el("td", "", item.accepted ? "accepted" : `rejected ${item.rejection_reason || ""}`));
    table.append(row);
  }
}

export function update(telemetry) {
  const ribbon = document.getElementById("nir-ribbon");
  const mode = telemetry && telemetry.nir && telemetry.nir.mode;
  if (ribbon) {
    ribbon.hidden = mode === "real";
    const label = mode === "off" ? "Infrared is off." : "Simulated infrared. Not a real camera.";
    if (ribbon.textContent !== label) ribbon.textContent = label;
  }
  const select = document.querySelector("#tab-nir select");
  if (!select || !telemetry) return;
  const ids = new Set((telemetry.tracks || []).map((track) => String(track.id)));
  for (const option of [...select.options]) {
    if (option.value && !ids.has(option.value)) option.remove();
  }
  for (const id of ids) {
    if (![...select.options].some((option) => option.value === id)) select.append(new Option(`Track ${id}`, id));
  }
}

export function hide() {}
