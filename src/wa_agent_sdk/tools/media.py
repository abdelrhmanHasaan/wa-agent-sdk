"""Image preparation for multimodal (vision) LLM input."""

from __future__ import annotations

import base64
import io

from ..exceptions import MediaError


def detect_image_mime(data: bytes) -> str | None:
    if data.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG"):
        return "image/png"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"BM"):
        return "image/bmp"
    return None


def prepare_image(
    data: bytes,
    *,
    max_side: int = 1600,
    keep_png_threshold: int = 512_000,
) -> tuple[str, str]:
    """Normalize raw image bytes into ``(base64, mime_type)`` for a vision model.

    Applies EXIF rotation, alpha flattening and a bounded downscale so large
    phone photos do not blow up provider payloads.
    """
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:  # pragma: no cover
        raise MediaError("Image handling requires 'pip install pillow'") from exc

    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:
        raise MediaError(f"Unreadable image ({exc})") from exc

    img = ImageOps.exif_transpose(img)
    original_format = (img.format or "").upper()

    width, height = img.size
    if max(width, height) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)

    out = io.BytesIO()
    if original_format == "PNG" and len(data) <= keep_png_threshold:
        img.save(out, format="PNG", optimize=True)
        mime = "image/png"
    else:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(out, format="JPEG", quality=88, optimize=True)
        mime = "image/jpeg"

    encoded = base64.b64encode(out.getvalue()).decode("ascii")
    return encoded, mime


def image_note_for_text_model() -> str:
    return "(The user sent an image, but the configured model cannot see images.)"
