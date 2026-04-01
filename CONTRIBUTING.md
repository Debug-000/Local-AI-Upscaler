# Contributing

Thanks for contributing.

## Setup

1. Create and activate a virtual environment.
2. Install `requirements.txt`.
3. Install `requirements-ai.txt` if you want to work on the AI path.
4. Download the model weights into `models/` locally. Do not commit them.

## Guidelines

- Keep the app local-first and easy to run.
- Prefer small, readable changes over broad refactors.
- Do not commit `.venv`, generated outputs, caches, or model weights.
- Keep Tkinter UI changes consistent with the existing visual direction.
- Preserve fallback behavior when the AI stack is unavailable.

## Pull requests

- Describe what changed and why.
- Include screenshots for UI changes.
- Mention any platform-specific caveats.
- Note whether you tested the fallback path, AI path, or both.
