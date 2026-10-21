export const bus = new EventTarget();
const TOKEN_KEY = "cineguard.token";

export function token() {
  return sessionStorage.getItem(TOKEN_KEY) || "";
}

export function withToken(url) {
  if (!url || !token()) return url;
  const join = url.includes("?") ? "&" : "?";
  return `${url}${join}token=${encodeURIComponent(token())}`;
}

export function setToken(value) {
  if (value) sessionStorage.setItem(TOKEN_KEY, value);
  else sessionStorage.removeItem(TOKEN_KEY);
}

export async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (token()) headers.set("Authorization", `Bearer ${token()}`);
  const init = { method: options.method || "GET", headers };
  if (options.json !== undefined) {
    headers.set("Content-Type", "application/json");
    init.body = JSON.stringify(options.json);
  }
  const response = await fetch(path, init);
  const text = await response.text();
  let data = null;
  if (text) {
    try { data = JSON.parse(text); } catch { data = text; }
  }
  if (!response.ok) {
    const detail = data && data.detail !== undefined ? data.detail : response.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return data;
}

export function stateText(state) {
  return {
    IDLE: "Idle",
    SUSPICIOUS: "Suspicious",
    ALERT_PENDING: "Alert pending",
    PIRACY_ALERT: "Piracy alert",
  }[state] || "Unknown";
}

export function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

export function clear(node) {
  node.replaceChildren();
}

export function field(label, control) {
  const wrap = el("label");
  wrap.append(document.createTextNode(label), control);
  return wrap;
}
