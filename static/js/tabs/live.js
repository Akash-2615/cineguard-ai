import { api, el, field, stateText } from "../api.js";
import { drawBars } from "../charts.js";
import { drawOverlay, hitTrack } from "../overlay.js";

const flags = { boxes: true, skeleton: true, glints: true, hand: true, seats: true, scores: true };
const rows = new Map();
let selected = null;
let last = null;
let video = null;
let canvas = null;
let why = null;
let bars = null;
let timeline = null;
const history = [];

export function mount(root) {
  root.className = "grid live-grid";
  const stageWrap = el("div", "panel");
  const stage = el("div", "stage");
  video = el("img");
  video.id = "live-stream";
  video.alt = "Live camera";
  video.src = "/api/v1/stream";
  canvas = el("canvas");
  canvas.setAttribute("aria-hidden", "true");
  stage.append(video, canvas);
  const toggles = el("div", "toggles");
  for (const [key, label] of [["boxes", "Boxes"], ["skeleton", "Skeleton"], ["glints", "Glints"], ["hand", "Hand ROI"], ["seats", "Seat labels"], ["scores", "Score labels"]]) {
    const input = el("input");
    input.type = "checkbox";
    input.checked = true;
    input.addEventListener("change", () => { flags[key] = input.checked; if (last) drawOverlay(canvas, video, last, flags); });
    const labelNode = field(label, input);
    labelNode.style.flexDirection = "row";
    toggles.append(labelNode);
  }
  stageWrap.append(stage, toggles, sourceControls(), thresholdControls());
  const side = el("div", "panel");
  side.append(el("h2", "", "Tracks"));
  const list = el("div");
  list.id = "track-list";
  why = el("div", "drawer");
  why.append(el("p", "empty", "Click a person for the score breakdown."));
  bars = el("canvas");
  timeline = el("p");
  side.append(list, why, bars, timeline);
  root.append(stageWrap, side);
  canvas.addEventListener("click", (event) => select(hitTrack(video, last, event)));
  new ResizeObserver(() => { if (last) drawOverlay(canvas, video, last, flags); }).observe(video);
}

function sourceControls() {
  const box = el("div", "controls");
  const kind = el("select");
  for (const item of ["file", "webcam", "rtsp"]) kind.append(new Option(item, item));
  const uri = el("select");
  uri.id = "clip-list";
  const custom = el("input");
  custom.placeholder = "webcam index, rtsp, or path";
  const loop = el("input");
  loop.type = "checkbox";
  loop.checked = true;
  const speed = el("input");
  speed.type = "range"; speed.min = "0.25"; speed.max = "4"; speed.step = "0.25"; speed.value = "1";
  const seek = el("input");
  seek.type = "range"; seek.min = "0"; seek.max = "1"; seek.step = "0.1"; seek.value = "0";
  const open = el("button", "", "Open source");
  open.type = "button";
  open.addEventListener("click", async () => {
    const value = custom.value.trim() || uri.value;
    await api("/api/v1/source", { method: "POST", json: { type: kind.value, uri: value, loop: loop.checked, speed: Number(speed.value) } });
  });
  for (const [action, label] of [["pause", "Pause"], ["play", "Play"], ["step", "Step"]]) {
    const button = el("button", "", label);
    button.type = "button";
    button.addEventListener("click", () => api("/api/v1/source/control", { method: "POST", json: { action } }));
    box.append(button);
  }
  seek.addEventListener("change", () => api("/api/v1/source/control", { method: "POST", json: { action: "seek", value: Number(seek.value) } }));
  box.append(field("Type", kind), field("Demo clip", uri), field("URI", custom), field("Loop", loop), field("Speed", speed), field("Seek seconds", seek), open);
  api("/api/v1/demo_clips").then((data) => {
    for (const name of data.files || []) uri.append(new Option(name, `demo_clips/${name}`));
    if (!(data.files || []).length) uri.append(new Option("No clips yet", ""));
  }).catch(() => uri.append(new Option("Clip list failed", "")));
  return box;
}

function thresholdControls() {
  const box = el("div", "controls");
  const specs = [
    ["piracy_threshold_sec", "Threshold seconds", 1, 120, 1, 30],
    ["conf_threshold", "Pose confidence", 0, 1, 0.05, 0.65],
    ["weight_pose", "Pose weight", 0, 1, 0.05, 0.25],
    ["weight_sightline", "Sightline weight", 0, 1, 0.05, 0.15],
    ["weight_glint", "Glint weight", 0, 1, 0.05, 0.35],
    ["weight_persistence", "Persistence weight", 0, 1, 0.05, 0.15],
    ["weight_stability", "Stability weight", 0, 1, 0.05, 0.1],
  ];
  const inputs = {};
  for (const [key, label, min, max, step, value] of specs) {
    const input = el("input");
    input.type = "range"; input.min = String(min); input.max = String(max); input.step = String(step); input.value = String(value);
    inputs[key] = input;
    input.addEventListener("change", () => api("/api/v1/config", { method: "POST", json: { [key]: Number(input.value) } }).catch(showError));
    box.append(field(label, input));
  }
  const reset = el("button", "", "Reset to defaults");
  reset.type = "button";
  reset.addEventListener("click", () => {
    const payload = {};
    for (const [key, , , , , value] of specs) {
      inputs[key].value = String(value);
      payload[key] = value;
    }
    api("/api/v1/config", { method: "POST", json: payload }).catch(showError);
  });
  box.append(reset);
  return box;
}

export function update(telemetry) {
  last = telemetry;
  if (video && canvas) drawOverlay(canvas, video, telemetry, flags);
  const list = document.getElementById("track-list");
  if (!list) return;
  const tracks = (telemetry.tracks || []).slice(0, 200);
  const seen = new Set();
  if (!tracks.length && !list.childElementCount) list.append(el("p", "empty", "No people tracked."));
  if (tracks.length) {
    const empty = list.querySelector(".empty");
    if (empty) empty.remove();
  }
  for (const track of tracks) {
    seen.add(track.id);
    let row = rows.get(track.id);
    if (!row) {
      row = { root: el("button", "row"), text: el("span") };
      row.root.type = "button";
      row.root.append(row.text);
      row.root.addEventListener("click", () => select(track));
      list.append(row.root);
      rows.set(track.id, row);
    }
    row.track = track;
    row.text.textContent = `#${track.id} ${stateText(track.state)} ${track.behavior || ""} ${track.seat || ""}`;
  }
  for (const [id, row] of rows) {
    if (!seen.has(id)) {
      row.root.remove();
      rows.delete(id);
    }
  }
  if (selected) {
    const fresh = tracks.find((track) => track.id === selected);
    if (fresh) showWhy(fresh);
  }
}

function select(track) {
  selected = track ? track.id : null;
  showWhy(track);
}

function showWhy(track) {
  why.replaceChildren();
  if (!track) {
    why.append(el("p", "empty", "Click a person for the score breakdown."));
    return;
  }
  why.append(el("h3", "", `Track ${track.id}: ${track.behavior || "idle"}`));
  why.append(el("p", "", track.behavior_reason || ""));
  why.append(el("p", "", `${stateText(track.state)} · ${track.seat || "Unmapped"}`));
  const breakdown = track.breakdown || {};
  drawBars(bars, ["pose", "sightline", "glint", "persistence", "stability"].map((key) => ({ label: key, value: Number(breakdown[key] || 0) })));
  history.push(`${track.id}:${track.state}`);
  if (history.length > 8) history.shift();
  timeline.textContent = `Recent states: ${history.join(" → ")}`;
  why.append(timeline);
}

export function videoElement() {
  return video;
}

function showError(error) {
  why.replaceChildren(el("p", "error", error.message));
}

export function togglePlayback() {
  const paused = last && last.source && last.source.paused;
  return api("/api/v1/source/control", { method: "POST", json: { action: paused ? "play" : "pause" } });
}
