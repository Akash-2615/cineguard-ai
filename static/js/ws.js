import { token } from "./api.js";

export function connectTelemetry(onMessage, onStatus) {
  let delay = 250;
  let socket = null;
  let stopped = false;

  function open() {
    if (stopped) return;
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const query = token() ? `?token=${encodeURIComponent(token())}` : "";
    socket = new WebSocket(`${proto}://${location.host}/ws/live_telemetry${query}`);
    socket.onopen = () => {
      delay = 250;
      onStatus("live");
    };
    socket.onmessage = (event) => {
      try {
        onMessage(JSON.parse(event.data));
      } catch (error) {
        onStatus("bad message");
      }
    };
    socket.onclose = () => {
      onStatus("reconnecting");
      if (!stopped) setTimeout(open, delay);
      delay = Math.min(8000, delay * 2);
    };
    socket.onerror = () => socket.close();
  }

  open();
  return () => {
    stopped = true;
    if (socket) socket.close();
  };
}
