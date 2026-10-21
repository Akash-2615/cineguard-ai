import { stateText } from "./api.js";

const BONES = [
  ["left_shoulder", "right_shoulder"],
  ["left_shoulder", "left_elbow"], ["left_elbow", "left_wrist"],
  ["right_shoulder", "right_elbow"], ["right_elbow", "right_wrist"],
  ["left_shoulder", "left_hip"], ["right_shoulder", "right_hip"],
  ["left_hip", "right_hip"],
  ["left_hip", "left_knee"], ["left_knee", "left_ankle"],
  ["right_hip", "right_knee"], ["right_knee", "right_ankle"],
  ["nose", "left_eye"], ["nose", "right_eye"],
  ["left_eye", "left_ear"], ["right_eye", "right_ear"],
];

function mediaBox(video) {
  const boxW = Math.max(1, video.clientWidth);
  const boxH = Math.max(1, video.clientHeight);
  const natW = video.naturalWidth || boxW;
  const natH = video.naturalHeight || boxH;
  const scale = Math.min(boxW / natW, boxH / natH);
  const w = natW * scale;
  const h = natH * scale;
  return { boxW, boxH, x: (boxW - w) / 2, y: (boxH - h) / 2, w, h };
}

export function drawOverlay(canvas, video, telemetry, flags) {
  const media = mediaBox(video);
  const width = media.boxW;
  const height = media.boxH;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  const px = (value) => media.x + value * media.w;
  const py = (value) => media.y + value * media.h;
  const tracks = (telemetry && telemetry.tracks) || [];
  for (const track of tracks) {
    const box = track.bbox || [0, 0, 0, 0];
    const x = px(box[0]);
    const y = py(box[1]);
    const w = (box[2] - box[0]) * media.w;
    const h = (box[3] - box[1]) * media.h;
    const alert = track.state === "PIRACY_ALERT";
    ctx.strokeStyle = alert ? "#ef4444" : "#f59e0b";
    ctx.lineWidth = 2;
    if (flags.boxes) ctx.strokeRect(x, y, w, h);
    if (flags.skeleton) drawBones(ctx, track.keypoints || {}, px, py);
    if (flags.hand && track.hand_bbox) {
      const hand = track.hand_bbox;
      ctx.strokeStyle = "#39f3ff";
      ctx.strokeRect(px(hand[0]), py(hand[1]), (hand[2] - hand[0]) * media.w, (hand[3] - hand[1]) * media.h);
    }
    if (flags.glints) {
      for (const glint of track.glints || []) {
        const g = glint.bbox;
        ctx.strokeStyle = glint.synthetic ? "#d56bff" : "#ffe56a";
        ctx.strokeRect(px(g[0]), py(g[1]), (g[2] - g[0]) * media.w, (g[3] - g[1]) * media.h);
      }
    }
    ctx.fillStyle = "#e7eef8";
    ctx.font = "12px sans-serif";
    const bits = [`#${track.id}`, stateText(track.state)];
    if (flags.seats && track.seat) bits.push(track.seat);
    if (flags.scores) bits.push(String(track.score));
    if (flags.boxes || flags.seats || flags.scores) ctx.fillText(bits.join(" "), x, Math.max(12, y - 4));
    if (flags.boxes) drawRing(ctx, x + 16, y + 16, track);
  }
}

function drawBones(ctx, points, px, py) {
  ctx.strokeStyle = "#7dffa8";
  ctx.beginPath();
  for (const [a, b] of BONES) {
    if (!points[a] || !points[b]) continue;
    ctx.moveTo(px(points[a][0]), py(points[a][1]));
    ctx.lineTo(px(points[b][0]), py(points[b][1]));
  }
  ctx.stroke();
}

function drawRing(ctx, x, y, track) {
  const progress = Math.max(0, Math.min(1, (track.active_s || 0) / (track.threshold_s || 1)));
  ctx.beginPath();
  ctx.strokeStyle = track.state === "PIRACY_ALERT" ? "#ef4444" : "#f59e0b";
  ctx.arc(x, y, 10, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * progress);
  ctx.stroke();
}

export function hitTrack(video, telemetry, event) {
  const rect = video.getBoundingClientRect();
  const media = mediaBox(video);
  const x = (event.clientX - rect.left - media.x) / media.w;
  const y = (event.clientY - rect.top - media.y) / media.h;
  return ((telemetry && telemetry.tracks) || []).find((track) => {
    const box = track.bbox || [];
    return x >= box[0] && x <= box[2] && y >= box[1] && y <= box[3];
  }) || null;
}
