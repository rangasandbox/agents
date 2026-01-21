from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
import logging

from livekit.agents import APIConnectOptions, APIError, APIStatusError, tts, utils
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

logger = logging.getLogger("indic-http-tts")

@dataclass
class _RequestConfig:
    url: str
    voice: str
    sample_rate: int
    num_channels: int
    output_dir: Path
    log_dir: Path


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
        self._file_token = utils.shortuuid()

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        # Give generous room for long responses; avoid per-read timeouts mid-stream.
        timeout_val = self._conn_options.timeout if self._conn_options.timeout is not None else 300
        timeout = aiohttp.ClientTimeout(
            total=None, sock_connect=timeout_val, sock_read=timeout_val
        )

        headers: dict[str, Any] = {}
        payload = {"voice": self._cfg.voice, "text": self._input_text, "stream": True}

        logger.info(
            "sending TTS request",
            extra={"voice": self._cfg.voice, "text": self._input_text, "url": self._cfg.url},
        )

        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                wav_path: Path | None = None
                log_path: Path | None = None
                file_handle = None

                async with session.post(self._cfg.url, json=payload, headers=headers) as resp:
                    if resp.status >= 400:
                        raise APIStatusError(resp.status, await resp.text())

                    # Stream the WAV response incrementally instead of buffering it all.
                    buffer = bytearray()
                    initialized = False
                    total_bytes = 0
                    request_id = resp.headers.get("x-request-id", utils.shortuuid())
                    # Ensure output directories exist before writing anything.
                    self._cfg.output_dir.mkdir(parents=True, exist_ok=True)
                    self._cfg.log_dir.mkdir(parents=True, exist_ok=True)

                    async for chunk in resp.content.iter_chunked(4096):
                        if not chunk:
                            continue

                        total_bytes += len(chunk)

                        if not initialized:
                            buffer.extend(chunk)

                            # wait until we have the full WAV header before initializing
                            if len(buffer) < 44:
                                continue

                            if buffer[0:4] != b"RIFF" or buffer[8:12] != b"WAVE":
                                raise APIError("indic_http_tts returned non-wav data or empty body")

                            output_emitter.initialize(
                                request_id=request_id,
                                sample_rate=self._cfg.sample_rate,
                                num_channels=self._cfg.num_channels,
                                mime_type="audio/wav",
                            )
                            wav_path = self._cfg.output_dir / f"resp_{self._file_token}.wav"
                            file_handle = wav_path.open("wb")
                            file_handle.write(buffer)
                            log_path = self._cfg.log_dir / f"resp_{self._file_token}.txt"

                            initialized = True
                            output_emitter.push(bytes(buffer))
                            logger.info(
                                "indic_http_tts stream started",
                                extra={
                                    "request_id": request_id,
                                    "first_chunk_bytes": len(buffer),
                                    "audio_path": str(wav_path),
                                    "log_path": str(log_path),
                                },
                            )
                            buffer.clear()
                            continue

                        output_emitter.push(chunk)
                        if file_handle is not None:
                            file_handle.write(chunk)
                        logger.info(
                            "indic_http_tts chunk",
                            extra={
                                "request_id": request_id,
                                "chunk_bytes": len(chunk),
                                "total_bytes": total_bytes,
                            },
                        )

                    if not initialized:
                        raise APIError("indic_http_tts returned non-wav data or empty body")

                    output_emitter.flush()
                    if file_handle is not None:
                        file_handle.flush()
                        file_handle.close()

                    if log_path is not None:
                        log_content = [
                            f"request_id={request_id}",
                            f"text={self._input_text}",
                            f"voice={self._cfg.voice}",
                            f"url={self._cfg.url}",
                            f"sample_rate={self._cfg.sample_rate}",
                            f"num_channels={self._cfg.num_channels}",
                            f"audio_path={wav_path}",
                            f"total_bytes={total_bytes}",
                        ]
                        log_path.write_text("\n".join(log_content))
                    logger.info(
                        "indic_http_tts stream completed",
                        extra={"request_id": request_id, "total_bytes": total_bytes},
                    )

            except asyncio.TimeoutError as e:
                # Treat as non-retryable so we don't spin retries on long streams
                raise APIError("indic_http_tts timeout", retryable=False) from e
            except aiohttp.ClientError as e:
                raise APIError(f"indic_http_tts network error: {e}") from e
            finally:
                if file_handle is not None and not file_handle.closed:
                    file_handle.close()
                # Explicitly signal no more audio to avoid downstream decoder writes after close
                try:
                    output_emitter.end_input()
                except RuntimeError:
                    # Emitter never started (e.g., connection failure before WAV header)
                    pass


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
        voice: str = "Priya",
        sample_rate: int = 24000,
        num_channels: int = 1,
        debug_audio_dir: str | Path = Path("responses/sub200_audio"),
        debug_log_dir: str | Path = Path("responses/sub200_logs"),
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=num_channels,
        )
        audio_dir = Path(debug_audio_dir)
        log_dir = Path(debug_log_dir)
        audio_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        self._cfg = _RequestConfig(
            url=url,
            voice=voice,
            sample_rate=sample_rate,
            num_channels=num_channels,
            output_dir=audio_dir,
            log_dir=log_dir,
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
