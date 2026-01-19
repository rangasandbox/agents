import logging
import os

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RunContext,
    cli,
    metrics,
    room_io,
)
from livekit.agents.llm import function_tool
from livekit.plugins import deepgram, silero, openai, google
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.agents.tts.stream_adapter import StreamAdapter
from indic_http_tts import IndicHTTPStreamingTTS

logger = logging.getLogger("indic-agent")

load_dotenv()



class MetricsTracker:
    """Lightweight tracker to mirror the implement_this example."""

    def __init__(self) -> None:
        self._events = []

    def collect(self, metrics_event) -> None:
        self._events.append(metrics_event)

    def print_session_summary(self) -> None:
        if not self._events:
            logger.info("No metrics collected for this session.")
            return
        logger.info("Session metrics collected:")
        for idx, event in enumerate(self._events, start=1):
            logger.info(f"[{idx}] {event}")


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="நீங்கள் ஒரு மென்மையான பேசும் தமிழ் பெண் உதவியாளர். "
            "நீங்கள் பயனர்களுடன் குரல் மூலம் தொடர்பு கொள்வீர்கள். "
            "உங்கள் பதில்கள் சுருக்கமாகவும், தெளிவாகவும் இருக்க வேண்டும். "
            "எமோஜி, நட்சத்திரங்கள், மார்க்டவுன் அல்லது பிற சிறப்பு எழுத்துக்களை உங்கள் பதில்களில் பயன்படுத்த வேண்டாம். "
            "நீங்கள் மரியாதையாகவும், நட்பாகவும் இருக்கிறீர்கள், மேலும் சில நேரங்களில் நகைச்சுவை உணர்வும் காட்டலாம். "
            "உங்களின் அனைத்து பதில்களும் தமிழ் மொழியிலேயே இருக்க வேண்டும், ஆனால் தேவையான சில சமயங்களில் English வார்த்தைகளை பயன்படுத்தலாம். "
            "பொதுவான உரையாடல் தமிழில் பேசுங்கள், மிகவும் formal அல்லது casual அல்லாமல்.",)

    async def on_enter(self):
        # when the agent is added to the session, it'll generate a reply
        # according to its instructions
        # Keep it uninterruptible so the client has time to calibrate AEC (Acoustic Echo Cancellation).
        self.session.generate_reply(allow_interruptions=False)

    # all functions annotated with @function_tool will be passed to the LLM when this
    # agent is active
    @function_tool
    async def lookup_weather(
        self, context: RunContext, location: str, latitude: str, longitude: str
    ):
        """Called when the user asks for weather related information.
        Ensure the user's location (city or region) is provided.
        When given a location, please estimate the latitude and longitude of the location and
        do not ask the user for them.

        Args:
            location: The location they are asking for
            latitude: The latitude of the location, do not ask user for it
            longitude: The longitude of the location, do not ask user for it
        """

        logger.info(f"Looking up weather for {location}")

        return "sunny with a temperature of 70 degrees."


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    # each log entry will include these fields
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }
    session = AgentSession(
        # Speech-to-text (STT) - Deepgram for Hindi
        stt=openai.STT(
            language="ta",  # Tamil language code
            detect_language=False,  # Explicitly set to Tamil
            model="gpt-4o-transcribe",
        ),
        # LLM - Google Gemini
        llm=google.LLM(
            model="gemini-2.0-flash",
            api_key=os.environ.get("GEMINI_API_KEY"),
        ),
        # TTS - Indic-19 with Aaradhya voice
        tts=StreamAdapter(
            tts=IndicHTTPStreamingTTS(
                url="http://tts.sub200.dev/indic-19/v1/tts/generate",
                voice="Abhiram",
            )
        ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        preemptive_generation=True,
        # sometimes background noise could interrupt the agent session, these are considered false positive interruptions
        resume_false_interruption=True,
        false_interruption_timeout=1.0,
    )

    # log metrics as they are emitted, and total usage after session is over
    usage_collector = metrics.UsageCollector()
    tracker = MetricsTracker()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)
        tracker.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")
        tracker.print_session_summary()

    # shutdown callbacks are triggered when the session is over
    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=MyAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(),
        ),
    )


if __name__ == "__main__":
    cli.run_app(server)
