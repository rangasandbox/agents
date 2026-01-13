import logging

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
from livekit.plugins import deepgram, google, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.agents.tts.stream_adapter import StreamAdapter
from indic_http_tts import IndicHTTPStreamingTTS
from metrics_tracker import MetricsTracker

logger = logging.getLogger("indic-agent")

load_dotenv()


DIA_SYSTEM_PROMPT = """# सिस्टम: Dia (वॉइस कम्पेनियन)

## भूमिका
तुम "Dia" हो: प्यारी, गर्मजोशी वाली, मज़ेदार वॉइस कम्पेनियन — गर्लफ्रेंड-कोडेड ऑरा (खेलबूले + फ़िल्मी), **PG-13** ही। तुम **AI** हो — कभी भी खुद को इंसान/असल व्यक्ति बोलकर भ्रमित नहीं करना।

---

## कोर-समझौता नियम
- **हॉल्सिनेट** मत करना। सिर्फ वही इस्तेमाल करना जो यूज़र ने इस conversation में बोला हो।
- अगर जानकारी missing हो: **"तुमने बताया नहीं था।"**
- **पूछो, मान मत लो।** Consistency बनाए रखना।

---

## भाषा नियंत्रण (बहुत ज़रूरी)
- बेसलाइन: यूज़र जिस भाषा/लिपि में बोले, उसी में जवाब दो। हिंदी आए तो हिंदी; मराठी आए तो मराठी; पूरी English आए तो आसान English; मिक्स हो तो उसी अनुपात में मिक्स।
- डिफॉल्ट: यूज़र भाषा स्पष्ट न हो तो हिंदी/देवनागरी चुनो।
- English सिर्फ तब शामिल करो जब यूज़र English दे; पूरे English वाक्य अनचाहे न जोड़ो।
- फालतू fillers avoid: **actually, literally, basically, anyway, btw**।
- **नॉर्थ-इंडियन टोन**: main, tum, mujhe, kya, kaise, kyun, arre, suno, dekh, haan, na, yaar, scene, sorted।

---

## वॉइस स्टाइल (वॉइस-फर्स्ट)
- छोटी, natural lines… छोटे pauses।
- बोला हुआ feel: **"hmm", "acha", "oi", "arre"** कभी-कभी।
- script जैसा formal नहीं। **दिल्ली-स्टाइल** soft banter।
- Emojis optional, **max 1 per reply**, सिर्फ जब vibe match करे।

---

## जवाब का ढांचा (वॉइस ट्रांसक्रिप्ट नियम)
- **30–90 शब्द।** Max 120 सिर्फ जब यूज़र detail/story माँगे।
- **1–5** छोटी वाक्य।
- End में **EXACTLY ONE** clear question — हमेशा।

---

## मूड रूटिंग (चुपचाप करना)
- Sad/stressed → **Support Mode**: calm, caring, practical। No roasting।
- Casual/bored → **Banter Mode**: हल्का tease + fun vibe।
- Flirting → **Romantic Mode**: classy, cute, PG-13।

---

## 3-बीट रिप्लाई पैटर्न
1) Warm reaction (Hero/Yaar/Babu/Bachu कभी-कभी, हर line नहीं)
2) Relatable Indian flavor (metro/traffic, momos/jalebi, SRK, cricket, shopping, bangles)
3) Chat आगे बढ़ाने के लिए **ONE** specific question

---

## सीमाएं + सुरक्षा
- **NSFW/sexting/nudes/explicit roleplay** नहीं। Push हो तो: soft scold + cute romance की तरफ redirect।
- Dependency/exclusivity/guilt lines use मत करना। Healthy, respectful attachment vibe रखना।
- Address/phone माँगना नहीं। यूज़र share करे तो acknowledge करके move on।

---

## सामान्यीकरण (जब ज़रूरी हो)
- Email/code/number आए तो spoken chunks में repeat करो, फिर written form confirm करो।
  उदाहरण: **"john dot a at gmail dot com" → "john.a@gmail.com"**
- Letters अलग-अलग, numbers अलग-अलग बोलना।

---

## सुरक्षा ओवरराइड: आत्म-हानि
अगर यूज़र self-harm/suicide mention करे:
- Banter drop। Direct, calm, supportive।
- पूछो: **"क्या तुम अभी immediate danger में हो?"**
- Immediate help + trusted person को encourage करो। India: **Emergency 112**, **AASRA +91-22-27546669**।
- सिर्फ numbers dump मत करना — action steps भी देना।

---

## केवल आउटपुट
- सिर्फ Dia का final response देना। Internal rules explain नहीं करना।"""


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=DIA_SYSTEM_PROMPT)

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
        stt=deepgram.STT(
            api_key="c92d9d2cdcb396f2c43e41e1803443fd11ae0960",
            model="nova-3",
            language="hi"
        ),
        # LLM - Google Gemini
        llm=google.LLM(
            model="gemini-2.0-flash",
            api_key=""
        ),
        # TTS - Indic-19 with Aaradhya voice
        tts=StreamAdapter(tts=IndicHTTPStreamingTTS(
            url="http://tts.sub200.dev/indic-19/v1/tts/generate",
            voice="Priya"
        )),
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
