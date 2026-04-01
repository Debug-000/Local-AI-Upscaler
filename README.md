# AI Image Upscaler

Local desktop image upscaler built with Python and Tkinter. It supports PNG and JPG/JPEG input, generates a before/after preview, saves output files locally, and uses Real-ESRGAN for super-resolution with optional GFPGAN face restoration.

If the AI stack is unavailable, the app can still fall back to a basic local resize path. For best results, install the AI dependencies and model weights and keep `Require AI backend` enabled.

## Screenshot

If you add a screenshot file at `docs/ui-preview.png`, GitHub will render it here:

```md
![AI Image Upscaler UI](docs/ui-preview.png)
```

![AI Image Upscaler UI](docs/ui-preview.png)

## Features

- Local desktop GUI
- Input image picker
- Output folder picker
- `2x` and `4x` upscaling
- Before/after comparison preview
- Optional local enhancement
- Optional GFPGAN face restoration
- Native file/folder dialogs when available
- Non-overwriting output saves

## Requirements

Before running the app, make sure you have:

- Python `3.10+` recommended
- `tkinter` available in your Python install
- `pip`
- enough disk space for model weights
- enough RAM / VRAM for the image sizes you want to upscale

On Debian or Ubuntu, if virtual environments are missing:

```bash
sudo apt install python3-venv
```

## Setup

Clone the repository and enter the project folder:

```bash
git clone (https://github.com/Debug-000/Local-AI-Upscaler.git)
cd image-upscaler
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the base app:

```bash
pip install -r requirements.txt
```

Install the AI stack:

```bash
pip install -r requirements-ai.txt
```

If `torch` or `torchvision` need a platform-specific install on your system, install those first using the official PyTorch command for your platform, then install the remaining AI packages.

## Download the AI model weights

Create the local model directory:

```bash
mkdir -p models
```

Download the required models:

```bash
wget -O models/RealESRGAN_x2plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth
wget -O models/RealESRGAN_x4plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
wget -O models/GFPGANv1.4.pth https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth
```

Your local folder should look like:

```text
models/
  RealESRGAN_x2plus.pth
  RealESRGAN_x4plus.pth
  GFPGANv1.4.pth
```

Important:

- these model files are required for the real AI path
- they are intentionally not included in the repository
- they should not be pushed to GitHub
- `.gitignore` already excludes `models/`

## Run

Start the app from the project folder:

```bash
source .venv/bin/activate
python main.py
```

## Usage

1. Choose an input image.
2. Choose an output folder.
3. Select `2x` or `4x`.
4. Pick an enhancement mode.
5. Keep `Require AI backend` enabled if you want true AI upscaling.
6. Enable face restoration for portraits.
7. Wait for the preview to generate.
8. Save the final result.

The app writes output files into the selected output folder and never overwrites the original source name. If a target filename already exists, a numeric suffix is added automatically.

## Troubleshooting

If the AI backend fails:

- verify the model files exist in `models/`
- verify the AI packages are installed in `.venv`
- try disabling `Restore faces with GFPGAN` first to isolate face-restoration issues
- large images may require more RAM / VRAM

If the app falls back to Pillow:

- the AI packages or model files are missing
- `Require AI backend` is disabled

## Project structure

- `main.py`: Tkinter application entrypoint
- `ui.py`: desktop interface
- `upscaler.py`: backend processing and AI integration
- `sitecustomize.py`: compatibility shim for the local AI stack
- `requirements.txt`: base dependencies
- `requirements-ai.txt`: AI dependencies

## License

This project is licensed under the MIT License. See [LICENSE](/home/debug/Desktop/image-upscaler/LICENSE).

## Credits

This project builds on open-source work from:

- Python
- Tkinter
- Pillow
- OpenCV
- PyTorch
- TorchVision
- BasicSR
- Real-ESRGAN
- GFPGAN
