import json
import os
import shutil
from pathlib import Path

# -----------------------
# CONFIG
# -----------------------
DATA_PATH = "../data/clip_feature_vectors_clustered.json"
IMG_DIR = "../images/original_images"
OUTPUT_DIR = "../data/viewer"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)

# -----------------------
# LOAD DATA
# -----------------------
with open(DATA_PATH) as f:
    data = json.load(f)

clusters = data["cluster"]
paths = data["paths"]
umap_2d = data["umap_2d"]

# copy images into viewer folder and build data
points = []
for fname, cluster, (x, y) in zip(paths, clusters, umap_2d):
    src = os.path.join(IMG_DIR, fname)
    if not os.path.exists(src):
        continue
    dst = os.path.join(OUTPUT_DIR, "images", fname)
    shutil.copy2(src, dst)
    points.append(
        {
            "filename": fname,
            "cluster": "noise" if cluster == -1 else f"cluster_{cluster}",
            "x": float(x),
            "y": float(y),
        }
    )

print(f"Copied {len(points)} images")

# -----------------------
# WRITE HTML
# -----------------------
html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Fashion Cluster Viewer</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: sans-serif; background: #111; color: #eee; display: flex; height: 100vh; overflow: hidden; }}

  #sidebar {{
    width: 260px;
    min-width: 260px;
    background: #1a1a1a;
    padding: 16px;
    overflow-y: auto;
    border-right: 1px solid #333;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }}

  #sidebar h2 {{ font-size: 14px; color: #aaa; text-transform: uppercase; letter-spacing: 1px; }}

  .cluster-btn {{
    padding: 8px 12px;
    border-radius: 6px;
    border: 2px solid transparent;
    cursor: pointer;
    font-size: 13px;
    text-align: left;
    width: 100%;
    color: white;
    opacity: 0.7;
    transition: opacity 0.2s;
  }}
  .cluster-btn:hover, .cluster-btn.active {{ opacity: 1; border: 2px solid white; }}

  #main {{ flex: 1; display: flex; flex-direction: column; overflow: hidden; }}

  #scatter-container {{
    height: 45%;
    position: relative;
    border-bottom: 1px solid #333;
    overflow: hidden;
    cursor: grab;
  }}
  #scatter-container:active {{ cursor: grabbing; }}

  canvas {{ position: absolute; top: 0; left: 0; }}

  #grid-container {{
    flex: 1;
    overflow-y: auto;
    padding: 16px;
  }}

  #grid-title {{ font-size: 13px; color: #aaa; margin-bottom: 12px; }}

  #grid {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }}

  .thumb {{
    width: 120px;
    height: 120px;
    object-fit: cover;
    border-radius: 6px;
    cursor: pointer;
    border: 2px solid transparent;
    transition: border 0.15s, transform 0.15s;
  }}
  .thumb:hover {{ transform: scale(1.05); border-color: white; }}

  #lightbox {{
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.85);
    z-index: 100;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    gap: 12px;
  }}
  #lightbox.show {{ display: flex; }}
  #lightbox img {{ max-height: 80vh; max-width: 80vw; border-radius: 8px; }}
  #lightbox-label {{ color: #eee; font-size: 13px; }}
  #lightbox-close {{
    position: absolute;
    top: 20px; right: 28px;
    font-size: 28px;
    cursor: pointer;
    color: white;
  }}

  #tooltip {{
    display: none;
    position: fixed;
    z-index: 50;
    pointer-events: none;
    flex-direction: column;
    align-items: center;
    gap: 4px;
  }}
  #tooltip img {{
    width: 140px;
    height: 140px;
    object-fit: cover;
    border-radius: 8px;
    border: 2px solid white;
    box-shadow: 0 4px 20px rgba(0,0,0,0.8);
  }}
  #tooltip-label {{
    font-size: 11px;
    color: #eee;
    background: rgba(0,0,0,0.7);
    padding: 3px 8px;
    border-radius: 4px;
  }}
</style>
</head>
<body>

<div id="sidebar">
  <h2>Clusters</h2>
  <button class="cluster-btn active" data-cluster="all" style="background:#444">All</button>
</div>

<div id="main">
  <div id="scatter-container">
    <canvas id="scatter"></canvas>
  </div>
  <div id="grid-container">
    <div id="grid-title">Click a cluster to filter</div>
    <div id="grid"></div>
  </div>
</div>

<div id="tooltip">
  <img id="tooltip-img" src="" />
  <div id="tooltip-label"></div>
</div>

<div id="lightbox">
  <span id="lightbox-close">&times;</span>
  <img id="lightbox-img" src="" />
  <div id="lightbox-label"></div>
</div>

<script>
const points = {json.dumps(points)};

const clusterNames = [...new Set(points.map(p => p.cluster))].sort();
const palette = [
<<<<<<< Updated upstream
  "#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6",
  "#1abc9c","#e67e22","#e91e63","#00bcd4","#8bc34a",
  "#ff5722","#607d8b","#673ab7","#009688","#ffc107"
=======
  "#e74c3c", // red
  "#3498db", // blue
  "#f1c40f", // yellow
  "#9b59b6", // purple
  "#e91e63", // pink
  "#795548", // brown
  "#8bc34a"  // green
>>>>>>> Stashed changes
];
const colorMap = {{}};
clusterNames.forEach((c, i) => {{
  colorMap[c] = c === "noise" ? "#555" : palette[i % palette.length];
}});

// sidebar buttons
const sidebar = document.getElementById("sidebar");
clusterNames.forEach(c => {{
  const btn = document.createElement("button");
  btn.className = "cluster-btn";
  btn.dataset.cluster = c;
  btn.style.background = colorMap[c];
  const count = points.filter(p => p.cluster === c).length;
  btn.textContent = `${{c}} (${{count}})`;
  sidebar.appendChild(btn);
}});

// state
let activeCluster = "all";
let scale = 1, offsetX = 0, offsetY = 0;
let dragging = false, lastX, lastY;

// scatter
const canvas = document.getElementById("scatter");
const ctx = canvas.getContext("2d");
const tooltip = document.getElementById("tooltip");
const tooltipImg = document.getElementById("tooltip-img");
const tooltipLabel = document.getElementById("tooltip-label");

function resizeCanvas() {{
  const container = document.getElementById("scatter-container");
  canvas.width = container.clientWidth;
  canvas.height = container.clientHeight;
  drawScatter();
}}

function getTransformedPoints() {{
  const xs = points.map(p => p.x);
  const ys = points.map(p => p.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const pad = 40;
  const w = canvas.width - pad * 2;
  const h = canvas.height - pad * 2;
  return points.map(p => ({{
    ...p,
    cx: pad + ((p.x - minX) / (maxX - minX)) * w,
    cy: pad + ((p.y - minY) / (maxY - minY)) * h,
  }}));
}}

let transformedPoints = [];

function drawScatter() {{
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  transformedPoints = getTransformedPoints();
  ctx.save();
  ctx.translate(offsetX, offsetY);
  ctx.scale(scale, scale);
  transformedPoints.forEach(p => {{
    const active = activeCluster === "all" || p.cluster === activeCluster;
    ctx.beginPath();
    ctx.arc(p.cx, p.cy, active ? 6 : 3, 0, Math.PI * 2);
    ctx.fillStyle = active ? colorMap[p.cluster] : "#333";
    ctx.globalAlpha = active ? 0.9 : 0.3;
    ctx.fill();
  }});
  ctx.restore();
  ctx.globalAlpha = 1;
}}

// find closest point to mouse on canvas
function getPointAt(mouseX, mouseY) {{
  const hitRadius = 12;
  let closest = null;
  let closestDist = Infinity;
  transformedPoints.forEach(p => {{
    const screenX = p.cx * scale + offsetX;
    const screenY = p.cy * scale + offsetY;
    const dist = Math.hypot(mouseX - screenX, mouseY - screenY);
    if (dist < hitRadius && dist < closestDist) {{
      closest = p;
      closestDist = dist;
    }}
  }});
  return closest;
}}

// hover — show tooltip
canvas.addEventListener("mousemove", e => {{
  if (dragging) return;
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;
  const p = getPointAt(mx, my);
  if (p) {{
    canvas.style.cursor = "pointer";
    tooltipImg.src = `images/${{p.filename}}`;
    tooltipLabel.textContent = `${{p.filename}} — ${{p.cluster}}`;
    tooltip.style.display = "flex";
    // position tooltip above cursor
    const tx = e.clientX - 70;
    const ty = e.clientY - 170;
    tooltip.style.left = tx + "px";
    tooltip.style.top = Math.max(8, ty) + "px";
  }} else {{
    canvas.style.cursor = dragging ? "grabbing" : "grab";
    tooltip.style.display = "none";
  }}
}});

canvas.addEventListener("mouseleave", () => {{
  tooltip.style.display = "none";
}});

// click on scatter point — open lightbox
canvas.addEventListener("click", e => {{
  if (dragging) return;
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;
  const p = getPointAt(mx, my);
  if (p) openLightbox(p);
}});

// grid
function renderGrid(cluster) {{
  const grid = document.getElementById("grid");
  const title = document.getElementById("grid-title");
  grid.innerHTML = "";
  const filtered = cluster === "all" ? points : points.filter(p => p.cluster === cluster);
  title.textContent = `${{cluster === "all" ? "All clusters" : cluster}} — ${{filtered.length}} images`;
  filtered.forEach(p => {{
    const img = document.createElement("img");
    img.className = "thumb";
    img.src = `images/${{p.filename}}`;
    img.title = `${{p.filename}} — ${{p.cluster}}`;
    img.style.borderColor = colorMap[p.cluster];
    img.addEventListener("click", () => openLightbox(p));
    grid.appendChild(img);
  }});
}}

// lightbox
function openLightbox(p) {{
  document.getElementById("lightbox-img").src = `images/${{p.filename}}`;
  document.getElementById("lightbox-label").textContent = `${{p.filename}} — ${{p.cluster}}`;
  document.getElementById("lightbox").classList.add("show");
  tooltip.style.display = "none";
}}
document.getElementById("lightbox-close").addEventListener("click", () => {{
  document.getElementById("lightbox").classList.remove("show");
}});
document.getElementById("lightbox").addEventListener("click", e => {{
  if (e.target === document.getElementById("lightbox")) {{
    document.getElementById("lightbox").classList.remove("show");
  }}
}});

// cluster filter
document.querySelectorAll(".cluster-btn").forEach(btn => {{
  btn.addEventListener("click", () => {{
    document.querySelectorAll(".cluster-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    activeCluster = btn.dataset.cluster;
    drawScatter();
    renderGrid(activeCluster);
  }});
}});

// pan + zoom
const container = document.getElementById("scatter-container");
container.addEventListener("mousedown", e => {{ dragging = true; lastX = e.clientX; lastY = e.clientY; }});
window.addEventListener("mouseup", () => {{ dragging = false; }});
window.addEventListener("mousemove", e => {{
  if (!dragging) return;
  offsetX += e.clientX - lastX;
  offsetY += e.clientY - lastY;
  lastX = e.clientX; lastY = e.clientY;
  drawScatter();
}});
container.addEventListener("wheel", e => {{
  e.preventDefault();
  const delta = e.deltaY > 0 ? 0.9 : 1.1;
  scale *= delta;
  drawScatter();
}}, {{ passive: false }});

window.addEventListener("resize", resizeCanvas);
resizeCanvas();
renderGrid("all");
</script>
</body>
</html>"""

output_path = os.path.join(OUTPUT_DIR, "index.html")
with open(output_path, "w") as f:
    f.write(html)

print(f"Saved to {output_path}")
print(f"Open in browser: open '{output_path}'")
