# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opaque `encrypted_content` tokens for stateless Responses API replay.

Clients that run with ``store=false`` (e.g. Codex) hand server-generated
reasoning and compaction items back through the ``encrypted_content`` field.
vLLM serialises the item payload into a versioned, compressed, base64 token so
that any replica can reconstruct the item without server-side state. The token
is opaque but not confidential: the same reasoning text is also returned in
plain form unless ``include_reasoning=false``.
"""

import json
import zlib
from typing import Any

import pybase64 as base64

from vllm.exceptions import VLLMValidationError

_TOKEN_PREFIX = "vllm1:"


def encode_encrypted_content(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    return _TOKEN_PREFIX + base64.urlsafe_b64encode(zlib.compress(raw)).decode()


def decode_encrypted_content(
    token: str, *, expected_type: str, parameter: str = "input"
) -> dict[str, Any]:
    """Decode a token produced by :func:`encode_encrypted_content`.

    Raises:
        VLLMValidationError: if the token was not produced by vLLM, is
            corrupted, or carries an item of a different ``type``.
    """
    payload: Any = None
    if token.startswith(_TOKEN_PREFIX):
        try:
            payload = json.loads(
                zlib.decompress(base64.urlsafe_b64decode(token[len(_TOKEN_PREFIX) :]))
            )
        except (ValueError, zlib.error):
            payload = None
    if not isinstance(payload, dict) or payload.get("type") != expected_type:
        raise VLLMValidationError(
            f"Invalid encrypted_content for a {expected_type} item.",
            parameter=parameter,
        )
    return payload
