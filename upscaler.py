from __future__ import annotations

import shutil
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageChops, ImageEnhance, ImageFilter, UnidentifiedImageError

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
ENHANCEMENT_PRESETS = ("none", "balanced", "strong")
GFPGAN_MODEL_FILENAME = "GFPGANv1.4.pth"


class UpscalerError(Exception):
    """Raised when an image cannot be processed."""


@dataclass
class UpscaleResult:
    input_path: Path
    output_path: Path
    scale: int
    enhancement: str
    face_restored: bool
    original_size: tuple[int, int]
    output_size: tuple[int, int]
    engine: str


def is_supported_image(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def get_image_dimensions(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            return image.size
    except FileNotFoundError as exc:
        raise UpscalerError(f"Input file was not found: {path}") from exc
    except UnidentifiedImageError as exc:
        raise UpscalerError(f"Invalid image file: {path.name}") from exc
    except OSError as exc:
        raise UpscalerError(f"Unable to open image file: {path.name}") from exc


def list_missing_real_esrgan_bits(scale: int, model_dir: Path) -> list[str]:
    missing: list[str] = []

    try:
        _ensure_torchvision_compat()
        import torch  # noqa: F401
        from basicsr.archs.rrdbnet_arch import RRDBNet  # noqa: F401
        from realesrgan import RealESRGANer  # noqa: F401
    except ImportError:
        missing.append(
            "Install optional AI dependencies with: pip install realesrgan basicsr torch torchvision"
        )

    model_path = model_dir / _model_filename(scale)
    if not model_path.exists():
        missing.append(f"Missing model weights: {model_path}")

    return missing


def list_missing_gfpgan_bits(model_dir: Path) -> list[str]:
    missing: list[str] = []

    try:
        _ensure_torchvision_compat()
        from gfpgan import GFPGANer  # noqa: F401
    except ImportError:
        missing.append("Install optional face restoration dependency with: pip install gfpgan")

    model_path = model_dir / GFPGAN_MODEL_FILENAME
    if not model_path.exists():
        missing.append(f"Missing face restoration model weights: {model_path}")

    return missing


class ImageUpscaler:
    def __init__(self, model_dir: Path | None = None) -> None:
        self.model_dir = model_dir or Path(__file__).resolve().parent / "models"

    def upscale_image(
        self,
        input_path: Path,
        output_dir: Path,
        scale: int,
        enhancement: str = "none",
        face_restore: bool = False,
        require_ai: bool = True,
    ) -> UpscaleResult:
        output_path = self._build_output_path(
            input_path.expanduser().resolve(),
            output_dir.expanduser().resolve(),
            scale,
        )
        return self._process_image(
            input_path=input_path,
            output_path=output_path,
            scale=scale,
            enhancement=enhancement,
            face_restore=face_restore,
            require_ai=require_ai,
        )

    def upscale_to_path(
        self,
        input_path: Path,
        output_path: Path,
        scale: int,
        enhancement: str = "none",
        face_restore: bool = False,
        require_ai: bool = True,
    ) -> UpscaleResult:
        return self._process_image(
            input_path=input_path,
            output_path=output_path,
            scale=scale,
            enhancement=enhancement,
            face_restore=face_restore,
            require_ai=require_ai,
        )

    def promote_preview_output(self, preview_path: Path, final_dir: Path, input_path: Path, scale: int) -> Path:
        final_dir = final_dir.expanduser().resolve()
        input_path = input_path.expanduser().resolve()
        final_dir.mkdir(parents=True, exist_ok=True)
        destination = self._build_output_path(input_path, final_dir, scale)
        shutil.copy2(preview_path, destination)
        return destination

    def _process_image(
        self,
        input_path: Path,
        output_path: Path,
        scale: int,
        enhancement: str,
        face_restore: bool,
        require_ai: bool,
    ) -> UpscaleResult:
        input_path = input_path.expanduser().resolve()
        output_path = output_path.expanduser().resolve()
        enhancement = enhancement.lower()

        if scale not in (2, 4):
            raise UpscalerError("Scale must be 2x or 4x.")
        if enhancement not in ENHANCEMENT_PRESETS:
            raise UpscalerError(
                f"Enhancement must be one of: {', '.join(ENHANCEMENT_PRESETS)}."
            )
        if not input_path.exists():
            raise UpscalerError(f"Input image does not exist: {input_path}")
        if not is_supported_image(input_path):
            raise UpscalerError("Only PNG and JPG/JPEG images are supported.")

        original_size = get_image_dimensions(input_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        real_esrgan_errors = list_missing_real_esrgan_bits(scale, self.model_dir)
        gfpgan_errors = list_missing_gfpgan_bits(self.model_dir) if face_restore else []

        if face_restore and not gfpgan_errors and not real_esrgan_errors:
            try:
                self._upscale_with_gfpgan(input_path, output_path, scale)
                engine = "GFPGAN + Real-ESRGAN"
            except Exception as exc:  # pragma: no cover - runtime depends on optional stack
                if require_ai:
                    raise UpscalerError(f"AI face restoration failed at runtime: {exc}") from exc
                self._upscale_with_pillow(input_path, output_path, scale)
                engine = f"Pillow fallback (AI face restoration failed at runtime: {exc})"
        elif not real_esrgan_errors:
            try:
                self._upscale_with_real_esrgan(input_path, output_path, scale)
                engine = "Real-ESRGAN"
            except Exception as exc:  # pragma: no cover - runtime depends on optional stack
                if require_ai:
                    raise UpscalerError(f"AI upscaling failed at runtime: {exc}") from exc
                self._upscale_with_pillow(input_path, output_path, scale)
                engine = f"Pillow fallback (Real-ESRGAN unavailable at runtime: {exc})"
        else:
            if require_ai:
                raise UpscalerError(self._build_ai_missing_message(scale, face_restore))
            self._upscale_with_pillow(input_path, output_path, scale)
            engine = "Pillow fallback"

        if enhancement != "none":
            self._enhance_output_image(output_path, enhancement)

        output_size = get_image_dimensions(output_path)
        return UpscaleResult(
            input_path=input_path,
            output_path=output_path,
            scale=scale,
            enhancement=enhancement,
            face_restored=face_restore and engine == "GFPGAN + Real-ESRGAN",
            original_size=original_size,
            output_size=output_size,
            engine=engine,
        )

    def get_backend_status(self) -> str:
        lines = ["Super-resolution backend:"]
        for scale in (2, 4):
            missing = list_missing_real_esrgan_bits(scale, self.model_dir)
            if missing:
                lines.append(f"{scale}x AI backend not ready.")
                lines.extend(f"- {item}" for item in missing)
            else:
                lines.append(f"{scale}x AI backend ready.")
        lines.append("")
        lines.append("Face restoration backend:")
        gfpgan_missing = list_missing_gfpgan_bits(self.model_dir)
        if gfpgan_missing:
            lines.append("GFPGAN face restoration not ready.")
            lines.extend(f"- {item}" for item in gfpgan_missing)
        else:
            lines.append("GFPGAN face restoration ready.")
        lines.append("")
        lines.append("Fallback backend always available: Pillow resize plus optional local enhancement.")
        lines.append("Without the AI backend, enlarged images can still look soft or blurry.")
        return "\n".join(lines)

    def _upscale_with_pillow(self, input_path: Path, output_path: Path, scale: int) -> None:
        try:
            with Image.open(input_path) as image:
                source = image.copy()
                resized = image.resize(
                    (image.width * scale, image.height * scale),
                    resample=Image.Resampling.LANCZOS,
                )
                if source.mode in ("RGB", "RGBA", "L"):
                    # Blend in a second pass to recover some edge contrast after Lanczos resize.
                    bicubic = source.resize(
                        (image.width * scale, image.height * scale),
                        resample=Image.Resampling.BICUBIC,
                    )
                    resized = Image.blend(bicubic, resized, alpha=0.7)
                save_kwargs = {}
                if output_path.suffix.lower() in {".jpg", ".jpeg"}:
                    if resized.mode not in ("RGB", "L"):
                        resized = resized.convert("RGB")
                    save_kwargs["quality"] = 95
                resized.save(output_path, **save_kwargs)
        except UnidentifiedImageError as exc:
            raise UpscalerError(f"Invalid image file: {input_path.name}") from exc
        except OSError as exc:
            raise UpscalerError(f"Unable to process image: {exc}") from exc

    def _upscale_with_real_esrgan(self, input_path: Path, output_path: Path, scale: int) -> None:
        import cv2

        upsampler = self._build_realesrgan_upsampler(scale)
        image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise UpscalerError(f"Invalid image file: {input_path.name}")

        output, _ = upsampler.enhance(image, outscale=scale)
        if not cv2.imwrite(str(output_path), output):
            raise UpscalerError(f"Failed to write output image: {output_path}")

    def _upscale_with_gfpgan(self, input_path: Path, output_path: Path, scale: int) -> None:
        import cv2
        from gfpgan import GFPGANer

        upsampler = self._build_realesrgan_upsampler(scale)
        image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise UpscalerError(f"Invalid image file: {input_path.name}")

        restorer = GFPGANer(
            model_path=str(self.model_dir / GFPGAN_MODEL_FILENAME),
            upscale=scale,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=upsampler,
        )
        _, _, output = restorer.enhance(
            image,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
        )
        if not cv2.imwrite(str(output_path), output):
            raise UpscalerError(f"Failed to write output image: {output_path}")

    def _build_realesrgan_upsampler(self, scale: int):
        _ensure_torchvision_compat()
        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        model_path = self.model_dir / _model_filename(scale)
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=scale,
        )
        return RealESRGANer(
            scale=scale,
            model_path=str(model_path),
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=torch.cuda.is_available(),
        )

    def _enhance_output_image(self, output_path: Path, enhancement: str) -> None:
        try:
            with Image.open(output_path) as image:
                enhanced = self._apply_enhancement(image, enhancement)
                save_kwargs = {}
                if output_path.suffix.lower() in {".jpg", ".jpeg"}:
                    if enhanced.mode not in ("RGB", "L"):
                        enhanced = enhanced.convert("RGB")
                    save_kwargs["quality"] = 95
                enhanced.save(output_path, **save_kwargs)
        except UnidentifiedImageError as exc:
            raise UpscalerError(f"Invalid image file: {output_path.name}") from exc
        except OSError as exc:
            raise UpscalerError(f"Unable to enhance image: {exc}") from exc

    def _apply_enhancement(self, image: Image.Image, enhancement: str) -> Image.Image:
        if enhancement == "none":
            return image.copy()

        presets = {
            "balanced": {
                "blur_radius": 0.9,
                "detail_gain": 1.35,
                "sharpness": 1.22,
                "contrast": 1.06,
                "color": 1.02,
                "filter": ImageFilter.UnsharpMask(radius=1.3, percent=135, threshold=2),
            },
            "strong": {
                "blur_radius": 1.3,
                "detail_gain": 1.7,
                "sharpness": 1.4,
                "contrast": 1.12,
                "color": 1.03,
                "filter": ImageFilter.UnsharpMask(radius=1.8, percent=185, threshold=1),
            },
        }
        preset = presets[enhancement]

        result = image.copy()
        if result.mode not in ("RGB", "RGBA", "L"):
            result = result.convert("RGB")

        # High-pass style detail boost so the enhancer has a visible effect on fallback output.
        blurred = result.filter(ImageFilter.GaussianBlur(radius=preset["blur_radius"]))
        detail = ImageChops.subtract(result, blurred, scale=1.0, offset=128)
        detail = ImageEnhance.Contrast(detail).enhance(preset["detail_gain"])
        result = ImageChops.overlay(result, detail)
        result = result.filter(preset["filter"])
        result = ImageEnhance.Sharpness(result).enhance(preset["sharpness"])
        result = ImageEnhance.Contrast(result).enhance(preset["contrast"])
        result = ImageEnhance.Color(result).enhance(preset["color"])
        return result

    def _build_output_path(self, input_path: Path, output_dir: Path, scale: int) -> Path:
        stem = input_path.stem
        suffix = input_path.suffix.lower()
        candidate = output_dir / f"{stem}_upscaled_{scale}x{suffix}"
        counter = 1
        while candidate.exists():
            candidate = output_dir / f"{stem}_upscaled_{scale}x_{counter}{suffix}"
            counter += 1
        return candidate

    def _build_ai_missing_message(self, scale: int, face_restore: bool) -> str:
        lines = [f"AI backend is required for {scale}x upscaling, but it is not ready."]
        for item in list_missing_real_esrgan_bits(scale, self.model_dir):
            lines.append(f"- {item}")
        if face_restore:
            for item in list_missing_gfpgan_bits(self.model_dir):
                lines.append(f"- {item}")
        lines.append("Uncheck 'Require AI backend' in the app if you want basic resize fallback.")
        return "\n".join(lines)


def _ensure_torchvision_compat() -> None:
    module_name = "torchvision.transforms.functional_tensor"
    if module_name in sys.modules:
        return

    try:
        from torchvision.transforms.functional_tensor import rgb_to_grayscale  # noqa: F401
        return
    except ImportError:
        from torchvision.transforms.functional import rgb_to_grayscale

        shim = types.ModuleType(module_name)
        shim.rgb_to_grayscale = rgb_to_grayscale
        sys.modules[module_name] = shim


def _model_filename(scale: int) -> str:
    if scale == 2:
        return "RealESRGAN_x2plus.pth"
    if scale == 4:
        return "RealESRGAN_x4plus.pth"
    raise ValueError(f"Unsupported scale: {scale}")


def supported_formats_text() -> str:
    extensions: Iterable[str] = sorted(ext.lstrip(".").upper() for ext in SUPPORTED_EXTENSIONS)
    return ", ".join(extensions)


def enhancement_options() -> tuple[str, ...]:
    return ENHANCEMENT_PRESETS
