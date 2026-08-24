"""Terminal QR rendering for WhatsApp device linking."""

from __future__ import annotations

import io

import qrcode
from qrcode.constants import ERROR_CORRECT_M


def render_qr(data: str) -> str:
    """Render *data* as an ASCII QR code string suitable for any terminal."""
    qr = qrcode.QRCode(error_correction=ERROR_CORRECT_M, border=2, box_size=1)
    qr.add_data(data)
    qr.make(fit=True)
    buf = io.StringIO()
    qr.print_ascii(out=buf, tty=False, invert=True)
    return buf.getvalue()


def print_pairing_qr(qr_data: str) -> None:
    banner = " WhatsApp pairing "
    line = "=" * 56
    print()
    print(line)
    print(banner.center(len(line)))
    print("Open WhatsApp > Settings > Linked devices > Link a device".center(len(line)))
    print(line)
    print(render_qr(qr_data))
