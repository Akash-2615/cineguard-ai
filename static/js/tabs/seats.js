import { api, el, field, stateText } from "../api.js";

const SEAT_KEY = "cineguard.myseat";
const ACTIONS = {
  idle: "Idle",
  texting_like: "Texting",
  eating_like: "Eating",
  stretching_like: "Stretch",
  recording_candidate: "Phone up",
};

let grid = null;
let status = null;
let chosen = sessionStorage.getItem(SEAT_KEY) || "";
let frozen = null;
let points = [];
let calibration = null;
const recent = new Map();

export function mount(root) {
  root.className = "grid";
  status = el("p", "empty", "Click the seat you are in.");
  const house = el("div", "theatre panel");
  const screen = el("div", "screen", "SCREEN");
  screen.setAttribute("role", "img");
  screen.setAttribute("aria-label", "Screen, facing the camera");
  grid = el("div", "seat-grid");
  const lane = el("div", "camera-lane");
  lane.append(el("div", "beam"), el("div", "camera-spot", "CAMERA"));
  house.append(screen, grid, lane);
  const setup = el("details");
  setup.append(el("summary", "", "Calibrate corners"), wizard());
  root.append(house, status, setup);
  load();
}

async function load() {
  try {
    calibration = await api("/api/v1/calibration");
    drawGrid(calibration);
    status.textContent = chosen ? `You are in ${chosen}.` : "Click the seat you are in.";
    status.className = "";
  } catch (error) {
    status.textContent = error.message;
    status.className = "error";
  }
}

function drawGrid(model) {
  const rows = model.layout.rows;
  const seats = model.layout.seats_per_row;
  const labels = model.layout.row_labels || "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  grid.style.gridTemplateColumns = `repeat(${seats}, minmax(28px, 1fr))`;
  grid.replaceChildren();
  for (let row = 0; row < rows; row += 1) {
    for (let seat = 1; seat <= seats; seat += 1) {
      const number = model.layout.seat_numbering === "right_to_left" ? seats + 1 - seat : seat;
      const letter = labels[row] || String(row + 1);
      const name = `Row ${letter}, Seat ${number}${model.uncalibrated ? " (uncalibrated)" : ""}`;
      const button = el("button", "seat");
      button.type = "button";
      button.dataset.seat = name;
      button.append(el("span", "seat-id", `${letter}${number}`), el("span", "person"), el("span", "action", ""));
      button.addEventListener("click", () => choose(name));
      grid.append(button);
      if (name === chosen) button.classList.add("you");
    }
  }
}

function choose(name) {
  chosen = name;
  sessionStorage.setItem(SEAT_KEY, name);
  for (const button of grid.querySelectorAll(".seat")) {
    button.classList.toggle("you", button.dataset.seat === name);
  }
  if (status) status.textContent = `You are in ${name}.`;
}

export function update(telemetry) {
  if (!grid) return;
  const tracks = telemetry.tracks || [];
  const alerts = telemetry.active_alerts || telemetry.alerts || [];
  const placed = placeTracks(tracks);
  const seen = new Set(placed.keys());
  for (const alert of alerts) {
    if (!alert.seat) continue;
    seen.add(alert.seat);
    recent.set(alert.seat, alert.state === "PIRACY_ALERT" ? "alert" : "suspicious");
  }
  for (const [seat, kind] of recent) {
    if (!seen.has(seat) && kind !== "resolved") recent.set(seat, "resolved");
  }
  for (const button of grid.querySelectorAll(".seat")) {
    const name = button.dataset.seat;
    const track = placed.get(name);
    const kind = track
      ? (track.state === "PIRACY_ALERT" ? "alert" : track.state === "SUSPICIOUS" || track.state === "ALERT_PENDING" ? "suspicious" : "")
      : (recent.get(name) || "");
    button.classList.toggle("occupied", Boolean(track));
    button.classList.toggle("alert", kind === "alert");
    button.classList.toggle("suspicious", kind === "suspicious");
    button.classList.toggle("resolved", kind === "resolved" && !track);
    const action = button.querySelector(".action");
    if (action) action.textContent = track ? (ACTIONS[track.behavior] || stateText(track.state)) : "";
    const label = track
      ? `${name}. ${ACTIONS[track.behavior] || stateText(track.state)}.`
      : `${name}. ${button.classList.contains("you") ? "Your seat." : "Empty."}`;
    button.title = label;
    button.setAttribute("aria-label", label);
  }
  if (status && tracks.length && placed.size) {
    const mine = chosen && placed.get(chosen);
    status.textContent = mine
      ? `${ACTIONS[mine.behavior] || stateText(mine.state)} on your seat.`
      : "A person is on the map.";
  }
}

function placeTracks(tracks) {
  const buttons = [...grid.querySelectorAll(".seat")];
  const byName = new Map(buttons.map((button) => [button.dataset.seat, button]));
  const placed = new Map();
  const ranked = tracks.slice().sort((a, b) => Number(b.score || 0) - Number(a.score || 0));
  const primary = ranked[0];
  if (primary) {
    const preferred = (chosen && byName.has(chosen) && chosen)
      || (primary.seat && byName.has(primary.seat) && primary.seat)
      || (buttons[Math.floor(buttons.length / 2)] && buttons[Math.floor(buttons.length / 2)].dataset.seat);
    if (preferred) placed.set(preferred, primary);
  }
  for (const track of ranked.slice(1)) {
    const seat = track.seat && byName.has(track.seat) && !placed.has(track.seat) ? track.seat : freeSeat(buttons, placed);
    if (seat) placed.set(seat, track);
  }
  return placed;
}

function freeSeat(buttons, placed) {
  const open = buttons.find((button) => !placed.has(button.dataset.seat));
  return open ? open.dataset.seat : "";
}

function wizard() {
  const box = el("div", "panel");
  box.append(el("h2", "", "Calibration wizard"));
  const shot = el("canvas");
  shot.width = 640; shot.height = 360;
  const freeze = el("button", "", "Freeze frame");
  freeze.type = "button";
  freeze.addEventListener("click", () => {
    const video = document.getElementById("live-stream");
    if (!video) return;
    shot.width = video.naturalWidth || 640;
    shot.height = video.naturalHeight || 360;
    shot.getContext("2d").drawImage(video, 0, 0, shot.width, shot.height);
    frozen = true;
    points = [];
    hint.textContent = "Click the four corners: top-left, top-right, bottom-right, bottom-left.";
  });
  const hint = el("p", "empty", "Freeze a frame, then click four corners.");
  shot.addEventListener("click", async (event) => {
    const rect = shot.getBoundingClientRect();
    const x = (event.clientX - rect.left) / rect.width;
    const y = (event.clientY - rect.top) / rect.height;
    if (probeMode.checked) {
      try {
        const result = await api("/api/v1/calibration/probe", { method: "POST", json: { x: clamp01(x), y: clamp01(y) } });
        hint.textContent = result.label;
      } catch (error) {
        hint.textContent = error.message;
      }
      return;
    }
    if (!frozen || points.length >= 4) return;
    points.push([clamp01(x), clamp01(y)]);
    const ctx = shot.getContext("2d");
    ctx.fillStyle = "#22d3ee";
    ctx.beginPath();
    ctx.arc(x * shot.width, y * shot.height, 5, 0, Math.PI * 2);
    ctx.fill();
    hint.textContent = `${points.length} of 4 corners.`;
  });
  const rows = numInput(4);
  const seats = numInput(10);
  const numbering = el("select");
  numbering.append(new Option("left to right", "left_to_right"), new Option("right to left", "right_to_left"));
  const aisles = el("input");
  aisles.placeholder = "aisle after seats, e.g. 5";
  const name = el("input");
  name.value = "Main hall";
  const probeMode = el("input");
  probeMode.type = "checkbox";
  const save = el("button", "", "Save calibration");
  save.type = "button";
  save.addEventListener("click", async () => {
    if (points.length !== 4) {
      hint.textContent = "Four corners are required.";
      return;
    }
    const body = {
      name: name.value || "Main hall",
      camera_id: "cam-1",
      image_size: [shot.width, shot.height],
      image_points: points,
      tolerance: 0.08,
      layout: {
        rows: Number(rows.value),
        seats_per_row: Number(seats.value),
        row_labels: "ABCDEFGHIJKLMNOPQRSTUVWXYZ".slice(0, Number(rows.value)),
        seat_numbering: numbering.value,
        aisle_after_seats: aisles.value.split(",").map((item) => parseInt(item.trim(), 10)).filter((item) => item > 0),
      },
    };
    try {
      await api("/api/v1/calibration", { method: "PUT", json: body });
      await load();
      hint.textContent = "Calibration saved.";
    } catch (error) {
      hint.textContent = error.message;
    }
  });
  box.append(freeze, shot, hint, field("Name", name), field("Rows", rows), field("Seats per row", seats), field("Numbering", numbering), field("Aisles", aisles), field("Probe tool", probeMode), save);
  return box;
}

function numInput(value) {
  const input = el("input");
  input.type = "number";
  input.min = "1";
  input.value = String(value);
  return input;
}

function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}
