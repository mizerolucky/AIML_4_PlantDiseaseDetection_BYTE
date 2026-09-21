/*
 * FloraLens browser inference.
 *
 * The model runs client-side through onnxruntime-web. Nothing leaves the tab.
 *
 * The heatmap is real Grad-CAM, not an approximation of it. Because the network
 * ends in global average pooling followed by a single linear layer, the channel
 * weights that Grad-CAM obtains by backpropagating the class score are exactly
 * the rows of the classifier weight matrix:
 *
 *     alpha_k = (1/HW) * sum_ij d(y_c)/d(A_kij) = W[c][k] / HW
 *     cam     = relu(sum_k alpha_k * A_k)
 *
 * The 1/HW factor is a positive constant and disappears when the map is
 * normalised to [0, 1], so this computes the same map an autograd
 * implementation would. src/verify_gradcam.py in the repository checks that
 * against a reference implementation over real test images.
 */

const SIZE = 224;          // model input resolution
const CAM_ALPHA = 0.45;    // heatmap opacity at full intensity
const LOW_CONFIDENCE = 0.60;

const SAMPLES = [
  { file: "samples/early_blight_1.jpg", label: "Early blight" },
  { file: "samples/early_blight_2.jpg", label: "Early blight" },
  { file: "samples/late_blight_1.jpg",  label: "Late blight" },
  { file: "samples/late_blight_2.jpg",  label: "Late blight" },
  { file: "samples/healthy_1.jpg",      label: "Healthy" },
  { file: "samples/healthy_2.jpg",      label: "Healthy" },
];

const el = (id) => document.getElementById(id);
const statusBox = el("status");
const statusText = el("statusText");

let session = null;
let meta = null;
let lastResult = null;

function setStatus(text, state) {
  statusText.textContent = text;
  statusBox.classList.remove("ready", "error");
  if (state) statusBox.classList.add(state);
}

/* ---------------------------------------------------------------- setup */

async function init() {
  try {
    const metaResponse = await fetch("model/meta.json");
    if (!metaResponse.ok) throw new Error(`meta.json: HTTP ${metaResponse.status}`);
    meta = await metaResponse.json();

    // The runtime is served from this origin, so point it at the local copies
    // instead of letting it guess a CDN path. This has to be an absolute URL:
    // the runtime loads its glue code with a dynamic import(), and a bare
    // relative path like "vendor/" is not a valid module specifier.
    ort.env.wasm.wasmPaths = new URL("vendor/", window.location.href).href;
    // Single-threaded: multi-threading needs cross-origin isolation headers,
    // and this model takes a few milliseconds either way.
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;

    session = await ort.InferenceSession.create("model/floralens.onnx", {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });

    setStatus("Model ready", "ready");
    buildSampleStrip();
  } catch (err) {
    console.error(err);
    setStatus("Could not load the model", "error");
  }
}

function buildSampleStrip() {
  const strip = el("sampleStrip");
  SAMPLES.forEach((s) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.title = `Test-set example: ${s.label}`;
    const img = document.createElement("img");
    img.src = s.file;
    img.alt = `Test-set leaf labelled ${s.label}`;
    img.loading = "lazy";
    btn.appendChild(img);
    btn.addEventListener("click", () => loadFromUrl(s.file));
    strip.appendChild(btn);
  });
}

/* ------------------------------------------------------------ preprocessing */

/** Draw an image into a 224x224 canvas and return the normalised NCHW tensor. */
function toTensor(image) {
  const canvas = el("inputCanvas");
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.clearRect(0, 0, SIZE, SIZE);
  ctx.drawImage(image, 0, 0, SIZE, SIZE);

  const { data } = ctx.getImageData(0, 0, SIZE, SIZE);
  const out = new Float32Array(3 * SIZE * SIZE);
  const { mean, std } = meta;
  const plane = SIZE * SIZE;

  for (let i = 0; i < plane; i++) {
    const p = i * 4;
    out[i]             = (data[p]     / 255 - mean[0]) / std[0];
    out[plane + i]     = (data[p + 1] / 255 - mean[1]) / std[1];
    out[2 * plane + i] = (data[p + 2] / 255 - mean[2]) / std[2];
  }
  return new ort.Tensor("float32", out, [1, 3, SIZE, SIZE]);
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

/* ---------------------------------------------------------------- Grad-CAM */

/**
 * cam[i][j] = relu( sum_k W[classIndex][k] * features[k][i][j] ), normalised.
 * `features` arrives flat in CHW order.
 */
function computeCam(features, dims, classIndex) {
  const [, channels, height, width] = dims;
  const weights = meta.classifier_weight[classIndex];
  const cam = new Float32Array(height * width);

  for (let k = 0; k < channels; k++) {
    const w = weights[k];
    if (w === 0) continue;
    const offset = k * height * width;
    for (let p = 0; p < cam.length; p++) cam[p] += w * features[offset + p];
  }

  let min = Infinity;
  let max = -Infinity;
  for (let p = 0; p < cam.length; p++) {
    cam[p] = Math.max(cam[p], 0);           // the ReLU in Grad-CAM
    if (cam[p] < min) min = cam[p];
    if (cam[p] > max) max = cam[p];
  }

  const range = max - min;
  if (range <= 1e-12) cam.fill(0);
  else for (let p = 0; p < cam.length; p++) cam[p] = (cam[p] - min) / range;

  return { cam, height, width };
}

/** Blue -> green -> red ramp. Mirrors cam_to_overlay() in src/gradcam.py. */
function ramp(v) {
  return [
    Math.min(Math.max(1.5 * v - 0.5, 0), 1),
    Math.min(Math.max(1.5 - Math.abs(3 * v - 1.5), 0), 1),
    Math.min(Math.max(1 - 1.5 * v, 0), 1),
  ];
}

/**
 * Paint the overlay. The CAM is 7x7, so it is drawn into a small canvas and
 * scaled up with the browser's own smoothing rather than nearest-neighbour
 * blocks. The CAM values themselves are the verified ones; only this display
 * interpolation is the browser's.
 */
function drawOverlay(camData) {
  const { cam, height, width } = camData;

  const small = document.createElement("canvas");
  small.width = width;
  small.height = height;
  const sctx = small.getContext("2d");
  const simg = sctx.createImageData(width, height);
  for (let p = 0; p < cam.length; p++) {
    const [r, g, b] = ramp(cam[p]);
    simg.data[p * 4]     = r * 255;
    simg.data[p * 4 + 1] = g * 255;
    simg.data[p * 4 + 2] = b * 255;
    simg.data[p * 4 + 3] = Math.round(cam[p] * CAM_ALPHA * 255);
  }
  sctx.putImageData(simg, 0, 0);

  const out = el("camCanvas");
  const octx = out.getContext("2d");
  octx.clearRect(0, 0, SIZE, SIZE);
  octx.drawImage(el("inputCanvas"), 0, 0);
  octx.imageSmoothingEnabled = true;
  octx.imageSmoothingQuality = "high";
  octx.drawImage(small, 0, 0, SIZE, SIZE);
}

/* ------------------------------------------------------------------ render */

function render(probs, classIndex) {
  const classes = meta.classes;
  const confidence = probs[classIndex];

  const verdict = el("verdict");
  verdict.innerHTML = "";

  const label = document.createElement("span");
  label.className = "label";
  label.textContent = classes[classIndex];

  const conf = document.createElement("span");
  conf.className = "conf";
  conf.textContent = `confidence ${(confidence * 100).toFixed(1)}%`;

  verdict.append(label, conf);

  if (confidence < LOW_CONFIDENCE) {
    const flag = document.createElement("span");
    flag.className = "flag";
    flag.textContent = "low confidence — this may not be a potato leaf";
    verdict.appendChild(flag);
  }

  const list = el("scoreList");
  list.innerHTML = "";
  classes
    .map((name, i) => ({ name, p: probs[i], i }))
    .sort((a, b) => b.p - a.p)
    .forEach(({ name, p, i }) => {
      const row = document.createElement("div");
      row.className = "score-row" + (i === classIndex ? "" : " dim");
      row.innerHTML =
        `<div class="score-head"><span class="name"></span>` +
        `<span class="val">${(p * 100).toFixed(2)}%</span></div>` +
        `<div class="score-track"><div class="score-fill"></div></div>`;
      row.querySelector(".name").textContent = name;
      row.querySelector(".score-fill").style.width = `${Math.max(p * 100, 0.6)}%`;
      list.appendChild(row);
    });

  el("results").hidden = false;
}

/* --------------------------------------------------------------- inference */

async function run(image) {
  if (!session) return;
  setStatus("Running…");

  try {
    const input = toTensor(image);
    const output = await session.run({ [session.inputNames[0]]: input });

    const logits = Array.from(output.logits.data);
    const probs = softmax(logits);
    const classIndex = probs.indexOf(Math.max(...probs));

    const feats = output.features;
    drawOverlay(computeCam(feats.data, feats.dims, classIndex));
    render(probs, classIndex);

    lastResult = { probs, classIndex };
    setStatus("Model ready", "ready");
    el("results").scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (err) {
    console.error(err);
    setStatus("Inference failed", "error");
  }
}

function loadFromFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    setStatus("That file is not an image", "error");
    return;
  }
  const reader = new FileReader();
  reader.onload = (e) => loadFromUrl(e.target.result);
  reader.readAsDataURL(file);
}

function loadFromUrl(url) {
  const img = new Image();
  img.crossOrigin = "anonymous";
  img.onload = () => run(img);
  img.onerror = () => setStatus("Could not read that image", "error");
  img.src = url;
}

/* ------------------------------------------------------------------ wiring */

const drop = el("drop");
const fileInput = el("fileInput");

el("browseBtn").addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });
drop.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", (e) => loadFromFile(e.target.files[0]));

["dragenter", "dragover"].forEach((type) =>
  drop.addEventListener(type, (e) => { e.preventDefault(); drop.classList.add("hover"); })
);
["dragleave", "drop"].forEach((type) =>
  drop.addEventListener(type, (e) => { e.preventDefault(); drop.classList.remove("hover"); })
);
drop.addEventListener("drop", (e) => loadFromFile(e.dataTransfer.files[0]));

window.addEventListener("paste", (e) => {
  const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
  if (item) loadFromFile(item.getAsFile());
});

el("resetBtn").addEventListener("click", () => {
  el("results").hidden = true;
  fileInput.value = "";
  window.scrollTo({ top: 0, behavior: "smooth" });
});

/** Compose the input, the overlay and the scores into one downloadable PNG. */
el("downloadBtn").addEventListener("click", () => {
  if (!lastResult) return;
  const { probs, classIndex } = lastResult;
  const pad = 16;
  const W = SIZE * 2 + pad * 3;
  const H = SIZE + 118;

  const c = document.createElement("canvas");
  c.width = W;
  c.height = H;
  const ctx = c.getContext("2d");

  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, W, H);
  ctx.drawImage(el("inputCanvas"), pad, 52);
  ctx.drawImage(el("camCanvas"), pad * 2 + SIZE, 52);

  ctx.fillStyle = "#1b2118";
  ctx.font = "600 19px system-ui, sans-serif";
  ctx.fillText(
    `${meta.classes[classIndex]} — ${(probs[classIndex] * 100).toFixed(1)}% confidence`,
    pad, 30
  );

  ctx.font = "12px system-ui, sans-serif";
  ctx.fillStyle = "#55604f";
  ctx.fillText("Input", pad, 46);
  ctx.fillText("Grad-CAM", pad * 2 + SIZE, 46);
  ctx.fillText(
    meta.classes.map((n, i) => `${n}: ${(probs[i] * 100).toFixed(2)}%`).join("    "),
    pad, SIZE + 78
  );
  ctx.fillText("FloraLens — potato leaves only; not a diagnosis.", pad, SIZE + 98);

  const a = document.createElement("a");
  a.download = "floralens-result.png";
  a.href = c.toDataURL("image/png");
  a.click();
});

/*
 * Test hook. scripts/verify_browser_gradcam.py drives a real browser, feeds a
 * known feature map into computeCam below, and compares the result against the
 * PyTorch autograd Grad-CAM for the same tensor.
 *
 * Exposing the functions is what makes that check meaningful: the script calls
 * this file's implementation rather than a Python transcription of it, so a
 * divergence between the two languages cannot slip through unnoticed.
 */
window.__floralens = { computeCam, softmax, ramp, get meta() { return meta; } };

init();
