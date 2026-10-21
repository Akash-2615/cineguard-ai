import { api, bus, el } from "./api.js";
import { connectTelemetry } from "./ws.js";
import * as live from "./tabs/live.js";
import * as nir from "./tabs/nir.js";
import * as seats from "./tabs/seats.js";
import * as incidents from "./tabs/incidents.js";
import * as scenarios from "./tabs/scenarios.js";
import * as system from "./tabs/system.js";

const tabs = { live, nir, seats, incidents, scenarios, system };
const names = ["live", "nir", "seats", "incidents", "scenarios", "system"];
let current = "live";
let pending = null;
let scheduled = false;
let audio = null;
let unlocked = false;
const seenAlerts = new Set();
const MUTE_KEY = "cineguard.mute";

const conn = document.getElementById("conn");
const sourceName = document.getElementById("source-name");
const fps = document.getElementById("fps");
const latency = document.getElementById("latency");
const badge = document.getElementById("nir-badge");
const degraded = document.getElementById("degraded");
const authWarn = document.getElementById("auth-warn");
const liveRegion = document.getElementById("live-region");
const toasts = document.getElementById("toasts");

for (const name of names) tabs[name].mount(document.getElementById(`tab-${name}`));

document.querySelectorAll(".tabs button").forEach((button) => {
  button.addEventListener("click", () => show(button.dataset.tab));
});

document.getElementById("theme").addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
  document.documentElement.dataset.theme = next;
  localStorage.setItem("cineguard.theme", next);
  document.getElementById("theme").textContent = next === "light" ? "Dark theme" : "Light theme";
});
const savedTheme = localStorage.getItem("cineguard.theme");
if (savedTheme) {
  document.documentElement.dataset.theme = savedTheme;
  document.getElementById("theme").textContent = savedTheme === "light" ? "Dark theme" : "Light theme";
}

const mute = document.getElementById("mute");
mute.addEventListener("click", () => setMuted(!muted()));
if (localStorage.getItem(MUTE_KEY) === "1") setMuted(true);

document.getElementById("help").addEventListener("click", () => document.getElementById("shortcuts").showModal());
document.getElementById("close-help").addEventListener("click", () => document.getElementById("shortcuts").close());
document.addEventListener("pointerdown", () => { unlocked = true; }, { once: true });

document.addEventListener("keydown", (event) => {
  if (event.target.matches("input, textarea, select")) return;
  if (event.key >= "1" && event.key <= "6") show(names[Number(event.key) - 1]);
  if (event.key === " ") { event.preventDefault(); live.togglePlayback(); }
  if (event.key === "f" || event.key === "F") {
    const video = live.videoElement();
    if (video) video.requestFullscreen().catch(() => {});
  }
  if (event.key === "m" || event.key === "M") setMuted(!muted());
  if (event.key === "?") document.getElementById("shortcuts").showModal();
});

bus.addEventListener("open-seat", () => show("incidents"));

api("/api/v1/config").then((config) => {
  authWarn.hidden = Boolean(config.auth_enabled);
}).catch(() => { authWarn.hidden = false; });

api("/health").then(paintHealth).catch((error) => {
  degraded.hidden = false;
  degraded.textContent = error.message;
});

connectTelemetry((frame) => {
  pending = frame;
  if (!scheduled) {
    scheduled = true;
    requestAnimationFrame(flush);
  }
}, (state) => {
  conn.textContent = state === "live" ? "Live" : "Reconnecting";
  conn.className = state === "live" ? "pill ok" : "pill warn";
});

function flush() {
  scheduled = false;
  const frame = pending;
  if (!frame) return;
  paintFrame(frame);
  tabs.live.update(frame);
  tabs.nir.update(frame);
  tabs.seats.update(frame);
  tabs.system.update(frame);
  noticeAlerts(frame);
}

function paintFrame(frame) {
  const system = frame.system || {};
  const source = frame.source || {};
  fps.textContent = `FPS ${Number(system.fps || 0).toFixed(1)}`;
  latency.textContent = `Latency ${system.latency_ms == null ? "--" : system.latency_ms} ms`;
  const cameraOn = source.state === "open" || source.state === "playing";
  sourceName.textContent = cameraOn ? "Camera is on" : `Camera ${source.state || "stopped"}`;
  const mode = (frame.nir && frame.nir.mode) || "simulated";
  badge.textContent = mode === "real" ? "NIR: REAL" : mode === "off" ? "NIR: OFF" : "NIR: SIMULATED";
  badge.className = mode === "real" ? "pill nir-real" : mode === "off" ? "pill nir-off" : "pill nir-sim";
  if (system.degraded) {
    degraded.hidden = false;
    const errors = Array.isArray(system.errors) ? system.errors.filter(Boolean).join(" | ") : "";
    degraded.textContent = errors ? `DEGRADED — ${errors}` : "DEGRADED";
  } else {
    degraded.hidden = true;
    degraded.textContent = "";
  }
}

function paintHealth(health) {
  if (!health.degraded) return;
  degraded.hidden = false;
  const errors = Array.isArray(health.errors) ? health.errors.filter(Boolean).join(" | ") : "";
  degraded.textContent = errors ? `DEGRADED — ${errors}` : "DEGRADED";
}

function noticeAlerts(frame) {
  for (const alert of frame.active_alerts || []) {
    if (alert.state !== "PIRACY_ALERT") continue;
    const key = `${alert.id}:${alert.started_at}`;
    if (seenAlerts.has(key)) continue;
    seenAlerts.add(key);
    const message = `Piracy alert ${alert.seat || "unmapped"}`;
    toast(message);
    liveRegion.textContent = message;
    if (unlocked && !muted()) beep();
  }
}

function toast(message) {
  const node = el("div", "toast", message);
  toasts.append(node);
  setTimeout(() => node.remove(), 4000);
}

function beep() {
  const context = audio || new AudioContext();
  audio = context;
  const osc = context.createOscillator();
  const gain = context.createGain();
  osc.frequency.value = 740;
  gain.gain.setValueAtTime(0.0001, context.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.08, context.currentTime + 0.02);
  gain.gain.exponentialRampToValueAtTime(0.0001, context.currentTime + 0.22);
  osc.connect(gain).connect(context.destination);
  osc.start();
  osc.stop(context.currentTime + 0.23);
}

function muted() {
  return localStorage.getItem(MUTE_KEY) === "1";
}

function setMuted(value) {
  localStorage.setItem(MUTE_KEY, value ? "1" : "0");
  mute.textContent = value ? "Unmute alerts" : "Mute alerts";
  mute.setAttribute("aria-pressed", value ? "true" : "false");
}

function show(name) {
  current = name;
  for (const item of names) {
    const panel = document.getElementById(`tab-${item}`);
    const button = document.querySelector(`.tabs button[data-tab="${item}"]`);
    const active = item === name;
    panel.hidden = !active;
    if (button) button.setAttribute("aria-selected", active ? "true" : "false");
    if (active && tabs[item].showTab) tabs[item].showTab();
    if (!active && tabs[item].hide) tabs[item].hide();
  }
}

show("live");
