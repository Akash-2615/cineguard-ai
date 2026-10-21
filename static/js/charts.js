export function drawSeries(canvas, series) {
  const ctx = canvas.getContext("2d");
  const width = canvas.width = canvas.clientWidth * devicePixelRatio || 300;
  const height = canvas.height = 80 * devicePixelRatio;
  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = "#64748b";
  ctx.strokeRect(0, 0, width, height);
  const colors = ["#22d3ee", "#f59e0b", "#22c55e", "#ef4444"];
  series.forEach((item, index) => {
    const values = item.values || [];
    if (!values.length) return;
    ctx.beginPath();
    ctx.strokeStyle = colors[index % colors.length];
    values.forEach((value, i) => {
      const x = (i / Math.max(values.length - 1, 1)) * (width - 8) + 4;
      const y = height - 4 - Math.max(0, Math.min(1, value)) * (height - 8);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });
}

export function drawBars(canvas, entries) {
  const ctx = canvas.getContext("2d");
  const width = canvas.width = canvas.clientWidth * devicePixelRatio || 300;
  const height = canvas.height = 90 * devicePixelRatio;
  ctx.clearRect(0, 0, width, height);
  const count = Math.max(entries.length, 1);
  entries.forEach((entry, index) => {
    const barWidth = width / count - 8;
    const barHeight = Math.max(0, Math.min(1, entry.value)) * (height - 16);
    const x = index * (width / count) + 4;
    ctx.fillStyle = "#22d3ee";
    ctx.fillRect(x, height - 8 - barHeight, barWidth, barHeight);
  });
}
