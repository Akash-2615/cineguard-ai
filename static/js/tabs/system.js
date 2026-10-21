import { api, el, field, setToken, token } from "../api.js";
import { drawSeries } from "../charts.js";

const cpu = [];
const mem = [];
const latency = [];
let logBox = null;
let healthBox = null;
let chart = null;
let configBox = null;
let timer = 0;

export function mount(root) {
  root.className = "grid";
  healthBox = el("div", "panel");
  healthBox.append(el("p", "", "Loading health…"));
  chart = el("canvas");
  configBox = el("div", "panel");
  const tokenInput = el("input");
  tokenInput.type = "password";
  tokenInput.value = token();
  tokenInput.autocomplete = "off";
  const saveToken = el("button", "", "Store token for this tab");
  saveToken.type = "button";
  saveToken.addEventListener("click", () => setToken(tokenInput.value.trim()));
  logBox = el("pre", "panel");
  logBox.textContent = "Loading logs…";
  const limits = el("div", "panel");
  limits.append(el("h2", "", "Limitations"));
  const list = el("ul");
  for (const item of [
    "NIR in this build is simulated unless a real mono camera is selected.",
    "There is no measured theatre accuracy in this proof of concept.",
    "The pose model must be present locally. It is not downloaded.",
    "Authentication is enforced only after a token is configured on the server.",
  ]) list.append(el("li", "", item));
  limits.append(list);
  root.append(el("h2", "", "System"), healthBox, chart, configBox, field("API token", tokenInput), saveToken, el("h3", "", "Log tail"), logBox, limits);
  refresh();
}

export function showTab() {
  window.clearInterval(timer);
  timer = window.setInterval(refresh, 2000);
  refresh();
}

export function hide() {
  window.clearInterval(timer);
}

async function refresh() {
  try {
    const health = await api("/health");
    healthBox.replaceChildren();
    healthBox.append(el("p", "", `Version ${health.version || "2.0.0"}`));
    healthBox.append(el("p", "", `Model ${health.model_loaded ? "loaded" : "unavailable"} · source ${health.source && health.source.state}`));
    healthBox.append(el("p", "", `Restarts ${health.restarts || 0} · supervisor ${health.supervisor_halted ? "halted" : "running"}`));
    const errors = health.errors || [];
    healthBox.append(el("p", errors.length ? "error" : "empty", errors.length ? errors.join(" | ") : "No errors."));
    const config = await api("/api/v1/config");
    renderConfig(config);
    const logs = await api("/api/v1/logs");
    logBox.textContent = (logs.lines || []).slice(-200).join("\n") || "No log lines yet.";
  } catch (error) {
    healthBox.replaceChildren(el("p", "error", error.message));
  }
}

export function update(telemetry) {
  const system = telemetry.system || {};
  cpu.push(Number(system.cpu_percent || 0) / 100);
  mem.push(Number(system.memory_percent || 0) / 100);
  latency.push(Math.min(1, Number(system.latency_ms || 0) / 200));
  if (cpu.length > 60) { cpu.shift(); mem.shift(); latency.shift(); }
  if (chart) drawSeries(chart, [{ values: cpu }, { values: mem }, { values: latency }]);
}

function renderConfig(config) {
  if (configBox.dataset.ready === "1") return;
  configBox.dataset.ready = "1";
  configBox.replaceChildren(el("h3", "", "Config"));
  const keys = ["piracy_threshold_sec", "conf_threshold", "target_fps", "nir_mode", "playback_speed", "loop_video"];
  const inputs = {};
  for (const key of keys) {
    const input = el("input");
    input.value = String(config[key]);
    inputs[key] = input;
    configBox.append(field(key, input));
  }
  const save = el("button", "", "Apply config");
  save.type = "button";
  save.addEventListener("click", async () => {
    const payload = {};
    for (const key of keys) {
      if (key === "loop_video") payload[key] = inputs[key].value === "true";
      else if (key === "nir_mode") payload[key] = inputs[key].value;
      else payload[key] = Number(inputs[key].value);
    }
    try {
      await api("/api/v1/config", { method: "POST", json: payload });
    } catch (error) {
      configBox.append(el("p", "error", error.message));
    }
  });
  configBox.append(save);
}
