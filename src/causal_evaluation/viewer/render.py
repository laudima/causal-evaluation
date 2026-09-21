"""Render the self-contained question-viewer HTML page."""

import json
from pathlib import Path
from typing import Any

from .causal_diagram import supported_topologies

_ASSETS = Path(__file__).parent / "assets"


def render_viewer_html(records: list[dict[str, Any]]) -> str:
    """Render the self-contained HTML viewer, one page for the whole sample."""
    template = (_ASSETS / "viewer.html").read_text(encoding="utf-8")
    css = (_ASSETS / "viewer.css").read_text(encoding="utf-8")
    js = (_ASSETS / "viewer.js").read_text(encoding="utf-8")

    data_json = json.dumps(records, ensure_ascii=False)
    supported_json = json.dumps(sorted(supported_topologies()))
    js = js.replace("__RECORDS__", data_json).replace("__SUPPORTED__", supported_json)

    return template.replace("__CSS__", css).replace("__JS__", js)
