import { api, el } from "../api.js";

let timer = 0;
let cards = null;
let table = null;

export function mount(root) {
  root.className = "grid";
  const run = el("button", "", "Run all");
  run.type = "button";
  run.addEventListener("click", async () => {
    run.disabled = true;
    try {
      await api("/api/v1/scenarios/run", { method: "POST", json: {} });
      poll();
    } catch (error) {
      table.replaceChildren(el("caption", "error", error.message));
      run.disabled = false;
    }
  });
  const report = el("a", "", "Open printable report");
  report.href = "/report";
  report.target = "_blank";
  cards = el("div", "cards");
  table = el("table");
  root.append(el("h2", "", "Scenarios"), run, report, cards, table);
  refresh();
}

function poll() {
  window.clearInterval(timer);
  timer = window.setInterval(refresh, 1000);
}

async function refresh() {
  try {
    const manifest = await api("/api/v1/scenarios/manifest");
    const job = await api("/api/v1/scenarios/results");
    renderManifest(manifest.clips || [], job);
    if (job.status !== "running") window.clearInterval(timer);
  } catch (error) {
    table.replaceChildren(el("caption", "error", error.message));
  }
}

function renderManifest(clips, job) {
  const results = (job.results && job.results.clips) || [];
  const byId = new Map(results.map((clip) => [clip.id, clip]));
  table.replaceChildren();
  const head = el("tr");
  for (const name of ["Clip", "Expected", "Description", "Result"]) head.append(el("th", "", name));
  table.append(head);
  if (!clips.length) table.append(el("caption", "empty", "The manifest is empty."));
  for (const clip of clips) {
    const row = el("tr");
    const found = byId.get(clip.id);
    const chip = el("span", found ? `chip ${found.passed ? "pass" : "fail"}` : "chip", found ? (found.passed ? "pass" : "fail") : job.status || "not run");
    row.append(el("td", "", clip.label || clip.id), el("td", "", clip.expected || ""), el("td", "", clip.description || ""), el("td"));
    row.lastChild.append(chip);
    if (found && found.error) row.lastChild.append(document.createTextNode(` ${found.error}`));
    table.append(row);
  }
  const progress = job.progress || {};
  cards.replaceChildren();
  const metrics = (job.results && job.results.metrics) || {};
  for (const [label, value] of [
    ["Status", `${job.status || "idle"} ${progress.completed || 0}/${progress.total || clips.length}`],
    ["Accuracy", show(metrics.accuracy)],
    ["Precision", show(metrics.precision)],
    ["Recall", show(metrics.recall)],
    ["Mean time to alert", show(metrics.mean_time_to_alert)],
    ["False alarms / hour", show(metrics.false_alarms_per_hour)],
  ]) {
    const card = el("div", "card");
    card.append(el("div", "muted", label), el("strong", "", String(value)));
    cards.append(card);
  }
}

function show(value) {
  return value === null || value === undefined ? "n/a" : value;
}

export function showTab() {
  refresh();
}

export function update() {}
