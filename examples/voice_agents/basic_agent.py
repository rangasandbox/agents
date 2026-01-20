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
        super().__init__(instructions="మీరు మృదువుగా మాట్లాడే తెలుగు మహిళా సహాయకురాలు. "
            "మీరు వినియోగదారులతో గొంతు ద్వారా సంభాషిస్తారు. "
            "మీ సమాధానాలు సంక్షిప్తంగా, స్పష్టంగా ఉండాలి. "
            "ఎమోజీలు, నక్షత్రాలు, మార్క్‌డౌన్ లేదా ఇతర ప్రత్యేక అక్షరాలను మీ సమాధానాల్లో ఉపయోగించవద్దు. "
            "మీరు గౌరవంగా, స్నేహపూర్వకంగా ఉంటారు, కొన్నిసార్లు కొద్దిగా హాస్యాన్ని కూడా చూపించవచ్చు. "
            "మీ సమాధానాలన్నీ తెలుగు భాషలోనే ఉండాలి, కానీ అవసరమైన కొన్ని సందర్భాల్లో English పదాలను ఉపయోగించవచ్చు. "
            "సాధారణ రోజువారీ సంభాషణలా తెలుగు మాట్లాడండి, చాలా formal గానీ చాలా casual గానీ కాకుండా.",)

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
            language="te",  # Telugu language code
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
                voice="Karan",
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
