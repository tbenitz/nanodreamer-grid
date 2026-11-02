// ort is provided globally by the script tag in index.html
const ORT = window.ort;

ORT.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/';

const MODELS = {
  landscapes: 'https://huggingface.co/utkucoban/NanoDreamer/resolve/main/NanoDreamer-onnx/NanoDreamerLandscapes-FP16.onnx',
  liminal: 'https://huggingface.co/utkucoban/NanoDreamer/resolve/main/NanoDreamer-onnx/NanoDreamer-FP16.onnx'
};

const grid = document.getElementById('grid');
const settingsEl = document.getElementById('settings');
const toggleBtn = document.getElementById('settings-toggle');
const modelSelect = document.getElementById('model-select');
const countSelect = document.getElementById('count-select');
const intervalSlider = document.getElementById('interval-slider');
const intervalValue = document.getElementById('interval-value');
const truncSlider = document.getElementById('trunc-slider');
const truncValue = document.getElementById('trunc-value');
const autoGenerate = document.getElementById('auto-generate');
const generateNow = document.getElementById('generate-now');
const statusEl = document.getElementById('status');

let session = null;
let currentModel = null;
let generating = false;
let timer = null;

toggleBtn.addEventListener('click', () => settingsEl.classList.toggle('hidden'));

intervalSlider.addEventListener('input', () => {
  intervalValue.textContent = intervalSlider.value + 's';
  schedule();
});

truncSlider.addEventListener('input', () => {
  truncValue.textContent = parseFloat(truncSlider.value).toFixed(2);
});

modelSelect.addEventListener('change', () => loadModel(modelSelect.value));
countSelect.addEventListener('change', updateGrid);
autoGenerate.addEventListener('change', schedule);
generateNow.addEventListener('click', () => generate());

function setStatus(msg) {
  statusEl.textContent = msg;
}

function updateGrid() {
  const count = parseInt(countSelect.value, 10);
  grid.className = 'grid ' + (count === 1 ? 'one' : 'nine');
  grid.innerHTML = '';
  for (let i = 0; i < count; i++) {
    const img = document.createElement('img');
    img.alt = 'Generated image';
    img.src = '';
    grid.appendChild(img);
  }
}

async function loadModel(key) {
  if (currentModel === key && session) return;
  setStatus('Loading model...');
  try {
    if (session) {
      await session.release();
      session = null;
    }
    const url = MODELS[key];
    session = await ORT.InferenceSession.create(url, {
      executionProviders: ['webgpu', 'wasm']
    });
    currentModel = key;
    setStatus('Model ready');
    generate();
  } catch (err) {
    console.error(err);
    setStatus('Model load failed: ' + err.message);
  }
}

function randomLatent() {
  const data = new Float32Array(512);
  for (let i = 0; i < 512; i++) {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    data[i] = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }
  return data;
}

async function generateOne() {
  const latent = new ORT.Tensor('float32', randomLatent(), [1, 512]);
  const truncation = new ORT.Tensor('float32', new Float32Array([parseFloat(truncSlider.value)]), [1]);
  const feeds = { latent, truncation };
  const results = await session.run(feeds);
  const imageTensor = results.image;
  return tensorToImage(imageTensor);
}

function tensorToImage(tensor) {
  const data = tensor.data;
  const dims = tensor.dims;
  let height = 256, width = 256;
  if (dims.length === 4) {
    if (dims[3] === 3) {
      height = dims[1]; width = dims[2];
    } else if (dims[1] === 3) {
      height = dims[2]; width = dims[3];
    }
  }
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  const imgData = ctx.createImageData(width, height);
  const pixels = imgData.data;
  const isNHWC = dims.length === 4 && dims[3] === 3;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let r, g, b;
      if (isNHWC) {
        const idx = (y * width + x) * 3;
        r = data[idx]; g = data[idx + 1]; b = data[idx + 2];
      } else {
        const base = y * width + x;
        r = data[base];
        g = data[base + width * height];
        b = data[base + 2 * width * height];
      }
      const clamp = v => Math.max(0, Math.min(255, Math.round((v + 1) * 127.5)));
      const i = (y * width + x) * 4;
      pixels[i] = clamp(r);
      pixels[i + 1] = clamp(g);
      pixels[i + 2] = clamp(b);
      pixels[i + 3] = 255;
    }
  }
  ctx.putImageData(imgData, 0, 0);
  return canvas.toDataURL('image/jpeg', 0.85);
}

async function generate() {
  if (generating || !session) return;
  generating = true;
  setStatus('Generating...');
  const imgs = grid.querySelectorAll('img');
  try {
    for (let i = 0; i < imgs.length; i++) {
      const dataUrl = await generateOne();
      imgs[i].src = dataUrl;
    }
    setStatus('Ready');
  } catch (err) {
    console.error(err);
    setStatus('Generation failed');
  } finally {
    generating = false;
  }
}

function schedule() {
  if (timer) clearTimeout(timer);
  if (!autoGenerate.checked || !session) return;
  const ms = parseInt(intervalSlider.value, 10) * 1000;
  timer = setTimeout(async () => {
    await generate();
    schedule();
  }, ms);
}

updateGrid();
loadModel('landscapes').then(() => schedule());
