# NanoDreamer Grid

Fullscreen, borderless grid viewer for the [NanoDreamer](https://huggingface.co/spaces/utkucoban/NanoDreamer) unconditional image models.

Runs entirely in the browser using WebGPU + ONNX Runtime Web. Choose between **Landscapes** and **Liminal Interiors**, display 1 or 9 images, and continuously generate new ones.

## Features
- Full-page grid (no borders)
- Model selector (Landscapes / Liminal Interiors)
- 1 or 9 images
- Auto-generation with adjustable interval
- Truncation control (affects quality/diversity)
- Settings overlay (toggle with ⚙)

## Usage
1. Open `index.html` in a modern browser that supports WebGPU (Chrome/Edge recommended).
2. Models (~48 MB each) are downloaded from Hugging Face on first load and cached by the browser.
3. Toggle settings to change options; generation continues automatically.

## Models
- Landscapes: `https://huggingface.co/utkucoban/NanoDreamer/resolve/main/NanoDreamer-onnx/NanoDreamerLandscapes-FP16.onnx`
- Liminal Interiors: `https://huggingface.co/utkucoban/NanoDreamer/resolve/main/NanoDreamer-onnx/NanoDreamer-FP16.onnx`

Credit to [utkucoban](https://huggingface.co/utkucoban) for the original NanoDreamer models and demo.
