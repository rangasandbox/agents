from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import aiohttp

from livekit.agents import APIConnectOptions, APIError, APIStatusError, tts, utils
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.utils.codecs import AudioStreamDecoder


def _infer_format_from_content_type(content_type: str | None) -> str | None:
    if not content_type:
        return None

    # content-type may look like audio/wav or audio/mpeg
    subtype = content_type.split("/")[-1]
    if "wav" in subtype:
        return "wav"
    if "mpeg" in subtype or "mp3" in subtype:
        return "mp3"
    return None


@dataclass
class _RequestConfig:
    url: str
    voice: str
    sample_rate: int
    num_channels: int


class _IndicChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        input_text: str,
        conn_options: APIConnectOptions,
        tts_parent: "IndicHTTPStreamingTTS",
        cfg: _RequestConfig,
    ) -> None:
        super().__init__(tts=tts_parent, input_text=input_text, conn_options=conn_options)
        self._tts_parent = tts_parent
        self._cfg = cfg

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        timeout_val = self._conn_options.timeout if self._conn_options.timeout is not None else 60
        timeout = aiohttp.ClientTimeout(total=timeout_val, sock_connect=timeout_val)

        headers: dict[str, Any] = {}
        payload = {"voice": self._cfg.voice, "text": self._input_text, "stream": True}

        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.post(self._cfg.url, json=payload, headers=headers) as resp:
                    if resp.status >= 400:
                        raise APIStatusError(resp.status, await resp.text())

                    # Treat response as a complete WAV payload (server returns audio/wav).
                    data = await resp.read()
                    if len(data) < 44 or data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
                        raise APIError("indic_http_tts returned non-wav data or empty body")

                    output_emitter.initialize(
                        request_id=resp.headers.get("x-request-id", utils.shortuuid()),
                        sample_rate=self._cfg.sample_rate,
                        num_channels=self._cfg.num_channels,
                        mime_type="audio/wav",
                    )

                    # send as a single segment
                    output_emitter.push(data)
                    output_emitter.flush()

            except asyncio.TimeoutError as e:
                raise APIError("indic_http_tts timeout") from e
            except aiohttp.ClientError as e:
                raise APIError(f"indic_http_tts network error: {e}") from e


class IndicHTTPStreamingTTS(tts.TTS):
    """
    Simple TTS wrapper that streams audio from the Indic HTTP endpoint.

    This keeps things minimal: it posts JSON to the provided URL and streams the
    binary audio response into LiveKit frames.
    """

    def __init__(
        self,
        *,
        url: str = "http://tts.sub200.dev/indic-19/v1/tts/generate",
        voice: str = "Kishan",
        sample_rate: int = 24000,
        num_channels: int = 1,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=num_channels,
        )
        self._cfg = _RequestConfig(
            url=url,
            voice=voice,
            sample_rate=sample_rate,
            num_channels=num_channels,
        )

    @property
    def model(self) -> str:
        return "indic-http"

    @property
    def provider(self) -> str:
        return self._cfg.url

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        # basic sentence pacing helps avoid long chunks if the TTS pauses mid-stream
        return _IndicChunkedStream(
            input_text=text,
            conn_options=conn_options,
            tts_parent=self,
            cfg=self._cfg,
        )
