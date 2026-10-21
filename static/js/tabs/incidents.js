import { api, bus, el, field, withToken } from "../api.js";
import { drawSeries } from "../charts.js";

let table = null;
let drawer = null;
let page = 0;
let seatFilter = "";
const limit = 20;

export function mount(root) {
  root.className = "grid";
  const filters = el("div", "filters");
  const status = el("select");
  status.append(new Option("Any status", ""));
  for (const name of ["open", "acknowledged", "false_positive", "confirmed", "expired"]) status.append(new Option(name, name));
  const seat = el("input");
  seat.placeholder = "Seat label";
  const since = el("input");
  since.type = "number"; since.placeholder = "Since timestamp";
  const until = el("input");
  until.type = "number"; until.placeholder = "Until timestamp";
  const synthetic = el("select");
  synthetic.append(new Option("All glints", ""), new Option("Synthetic only", "true"), new Option("Observed only", "false"));
  const apply = el("button", "", "Apply filters");
  apply.type = "button";
  apply.addEventListener("click", () => { page = 0; seatFilter = seat.value.trim(); load(status.value, since.value, until.value, synthetic.value); });
  const csv = el("a", "", "Export CSV");
  csv.href = "/api/v1/incidents/export.csv";
  filters.append(field("Status", status), field("Seat", seat), field("Since", since), field("Until", until), field("Synthetic", synthetic), apply, csv);
  table = el("table");
  drawer = el("div", "drawer");
  drawer.append(el("p", "empty", "Select an incident."));
  const pager = el("div", "row");
  const prev = el("button", "", "Previous");
  const next = el("button", "", "Next");
  prev.type = "button"; next.type = "button";
  prev.addEventListener("click", () => { page = Math.max(0, page - 1); load(status.value, since.value, until.value, synthetic.value); });
  next.addEventListener("click", () => { page += 1; load(status.value, since.value, until.value, synthetic.value); });
  pager.append(prev, next);
  root.append(el("h2", "", "Incidents"), filters, table, pager, drawer);
  bus.addEventListener("open-seat", (event) => {
    seat.value = event.detail;
    seatFilter = event.detail;
    page = 0;
    document.querySelector('[data-tab="incidents"]').click();
    load("", "", "", "");
  });
  load("", "", "", "");
}

async function load(status, since, until, synthetic) {
  const params = new URLSearchParams({ limit: String(limit), offset: String(page * limit) });
  if (status) params.set("status", status);
  if (seatFilter) params.set("seat", seatFilter);
  if (since) params.set("since", since);
  if (until) params.set("until", until);
  if (synthetic === "true" || synthetic === "false") params.set("synthetic", synthetic);
  table.replaceChildren(el("caption", "", "Loading incidents…"));
  try {
    const data = await api(`/api/v1/incidents?${params.toString()}`);
    render(data);
  } catch (error) {
    table.replaceChildren(el("caption", "error", error.message));
  }
}

function render(data) {
  table.replaceChildren();
  const head = el("tr");
  for (const name of ["ID", "Seat", "Status", "Score", "Synthetic", "Started"]) head.append(el("th", "", name));
  table.append(head);
  const items = data.items || [];
  if (!items.length) {
    table.append(el("caption", "empty", "No incidents match these filters."));
    return;
  }
  for (const item of items) {
    const row = el("tr");
    row.append(el("td", "", String(item.id)), el("td", "", item.seat || ""), el("td", "", item.status || ""), el("td", "", String(item.max_score)), el("td", "", item.synthetic ? "synthetic" : "observed"), el("td", "", String(item.started_ts)));
    row.addEventListener("click", () => openDetail(item.id));
    table.append(row);
  }
}

async function openDetail(id) {
  drawer.replaceChildren(el("p", "", "Loading…"));
  try {
    const item = await api(`/api/v1/incidents/${id}`);
    drawer.replaceChildren();
    drawer.append(el("h3", "", `Incident ${item.id}`));
    const image = el("img");
    image.alt = "Keyframe";
    image.src = withToken(item.keyframe_url);
    const video = el("video");
    video.controls = true;
    video.src = withToken(item.clip_url);
    const download = el("a", "", "Download clip");
    download.href = withToken(item.download_url || item.clip_url);
    const hash = el("p", "", `SHA-256 keyframe ${item.keyframe_sha256 || "missing"} · clip ${item.clip_sha256 || "missing"}`);
    const copy = el("button", "", "Copy hashes");
    copy.type = "button";
    copy.addEventListener("click", () => navigator.clipboard.writeText(`${item.keyframe_sha256 || ""} ${item.clip_sha256 || ""}`));
    const chart = el("canvas");
    const scores = (item.events || []).map((event) => Number(event.score || 0));
    drawer.append(image, video, download, hash, copy, chart, el("p", "", `Evidence ${item.evidence_status || "pending"} · ${item.nir_mode || ""} · ${item.evidence_type || ""}`));
    drawSeries(chart, [{ values: scores }]);
    const note = el("input");
    note.placeholder = "Reviewer note";
    const actions = el("div", "row");
    for (const status of ["acknowledged", "false_positive", "confirmed"]) {
      const button = el("button", "", status.replace("_", " "));
      button.type = "button";
      button.addEventListener("click", async () => {
        await api(`/api/v1/incidents/${id}/review`, { method: "POST", json: { status, note: note.value } });
        openDetail(id);
      });
      actions.append(button);
    }
    drawer.append(field("Note", note), actions);
  } catch (error) {
    drawer.replaceChildren(el("p", "error", error.message));
  }
}

export function update() {}
