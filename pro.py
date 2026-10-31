
import asyncio
import io
import os

import numpy as np
import soundfile as sf
import streamlit as st
from scipy.signal import resample_poly
from dotenv import load_dotenv

from agents import Agent
from agents.voice import (
    AudioInput,
    SingleAgentVoiceWorkflow,
    TTSModelSettings,
    VoicePipeline,
    VoicePipelineConfig,
)

# Load environment variables
load_dotenv()

SAMPLE_RATE = 24000

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Travel AI Voice Advisor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown(
    """
    <style>
    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
    );

    .stApp {
        background: linear-gradient(
            135deg,
            #080d1c 0%,
            #111b35 55%,
            #172554 100%
        );
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background: #0b1224;
        border-right: 1px solid #263453;
    }

    .hero {
        padding: 34px 28px;
        border-radius: 24px;
        background: linear-gradient(
            120deg,
            rgba(30, 64, 175, 0.65),
            rgba(109, 40, 217, 0.45),
            rgba(8, 145, 178, 0.30)
        );
        border: 1px solid rgba(148, 163, 184, 0.25);
        margin-bottom: 24px;
        animation: fadeIn 0.8s ease-in-out;
    }

    .hero h1 {
        color: white;
        font-size: clamp(30px, 4vw, 46px);
        font-weight: 800;
        margin-bottom: 10px;
    }

    .hero p {
        color: #dbeafe;
        font-size: 16px;
        line-height: 1.7;
    }

    .section-card {
        padding: 22px;
        border-radius: 18px;
        background: rgba(15, 23, 42, 0.78);
        border: 1px solid #293956;
        margin-bottom: 18px;
    }

    .section-title {
        color: #93c5fd;
        font-size: 21px;
        font-weight: 700;
        margin-bottom: 12px;
    }

    .status {
        display: inline-block;
        padding: 7px 13px;
        border-radius: 30px;
        background: rgba(16, 185, 129, 0.15);
        color: #6ee7b7;
        border: 1px solid rgba(16, 185, 129, 0.35);
        font-size: 13px;
        font-weight: 600;
    }

    .answer-card {
        padding: 24px;
        background: rgba(30, 41, 59, 0.85);
        border-radius: 18px;
        border: 1px solid #3b4d6c;
        line-height: 1.8;
        color: #f1f5f9;
        margin-top: 15px;
    }

    .feature {
        padding: 18px;
        min-height: 125px;
        border-radius: 16px;
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid #293956;
        margin-bottom: 12px;
    }

    .feature h3 {
        color: #bfdbfe;
        font-size: 17px;
    }

    .feature p {
        color: #cbd5e1;
        font-size: 13px;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 12px;
        min-height: 46px;
        font-weight: 700;
        border: 1px solid #60a5fa;
        transition: all 0.25s ease;
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        border-color: #a78bfa;
        box-shadow: 0 5px 22px rgba(96, 165, 250, 0.22);
    }

    div[data-testid="stAudioInput"] {
        border: 1px solid #3b4d6c;
        border-radius: 14px;
        padding: 12px;
        background: rgba(15, 23, 42, 0.7);
    }

    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(12px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "answer" not in st.session_state:
    st.session_state.answer = ""

if "audio_response" not in st.session_state:
    st.session_state.audio_response = None

if "question" not in st.session_state:
    st.session_state.question = ""

if "history" not in st.session_state:
    st.session_state.history = []


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown("## ✈️ Travel AI")
    st.caption("Your personal voice-powered travel companion")

    st.divider()

    api_key = st.text_input(
        "🔑 OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        placeholder="Enter your API key",
        help="Your key is used to run the voice agent.",
    )

    st.caption("Your API key is not displayed publicly.")

    model_name = st.selectbox(
        "🧠 AI Model",
        options=[
            "gpt-5.4-mini",
            "gpt-4o-mini",
            "gpt-4.1-mini",
        ],
        index=0,
    )

    voice_name = st.selectbox(
        "🔊 Assistant Voice",
        options=["coral", "alloy", "ash", "sage", "shimmer", "verse"],
        index=0,
    )

    st.divider()

    st.markdown("### 🌍 What I can help with")

    st.markdown(
        """
        - 🗺️ Destination suggestions
        - 🏨 Hotel and accommodation ideas
        - ✈️ Travel planning
        - 🎒 Packing recommendations
        - 🍜 Food and culture
        - 💰 Budget-friendly trips
        """
    )

    st.divider()

    if st.button("🗑️ Clear Conversation"):
        st.session_state.answer = ""
        st.session_state.audio_response = None
        st.session_state.question = ""
        st.session_state.history = []
        st.rerun()


# --------------------------------------------------
# HERO SECTION
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="status">● AI VOICE ASSISTANT</div>
        <h1>Explore the World with AI ✈️</h1>
        <p>
            Your intelligent Travel Advisor is ready to help.
            Ask your travel questions using your microphone,
            receive a natural AI response, and listen to the answer.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# FEATURE CARDS
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="feature">
            <h3>🎙️ Voice Input</h3>
            <p>Speak naturally using your device microphone.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="feature">
            <h3>🧠 AI Intelligence</h3>
            <p>Get friendly and concise travel advice.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div class="feature">
            <h3>🔊 Voice Output</h3>
            <p>Listen to your advisor's spoken response.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --------------------------------------------------
# MAIN VOICE INTERFACE
# --------------------------------------------------

st.markdown(
    '<div class="section-title">🎤 Talk to Your Travel Advisor</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <p style="color:#cbd5e1;">
        Click the microphone below, record your question, and submit it
        to receive an AI-generated voice response.
    </p>
    """,
    unsafe_allow_html=True,
)

audio_file = st.audio_input(
    "🎙️ Record your travel question",
    sample_rate=16000,
)

if audio_file is not None:
    st.audio(audio_file)

    if st.button("✨ Ask Travel Advisor", type="primary"):
        if not api_key.strip():
            st.error(
                "Please enter your OpenAI API key in the sidebar first."
            )

        else:
            try:
                # Set API key for the current run.
                os.environ["OPENAI_API_KEY"] = api_key.strip()

                # Read browser-recorded audio.
                audio_bytes = audio_file.getvalue()

                audio_array, input_rate = sf.read(
                    io.BytesIO(audio_bytes),
                    dtype="int16",
                    always_2d=True,
                )

                # Convert stereo audio to mono.
                mono_audio = audio_array.mean(axis=1).astype(np.int16)

                # Resample audio to 24 kHz if required.
                if input_rate != SAMPLE_RATE:
                    from math import gcd

                    common = gcd(SAMPLE_RATE, input_rate)

                    mono_audio = resample_poly(
                        mono_audio.astype(np.float32),
                        SAMPLE_RATE // common,
                        input_rate // common,
                    )

                    mono_audio = np.clip(
                        mono_audio,
                        -32768,
                        32767,
                    ).astype(np.int16)

                audio_input = AudioInput(buffer=mono_audio)

                # Create the agent using the selected model.
                agent = Agent(
                    name="Travel Advisor",
                    instructions=(
                        "You are a friendly and helpful travel advisor. "
                        "Answer the user's spoken travel question naturally. "
                        "Keep spoken answers to 2-3 sentences maximum. "
                        "Be clear, practical, and conversational. "
                        "Do not claim to have checked live prices, bookings, "
                        "or current availability unless you actually have."
                    ),
                    model=model_name,
                )

                # Configure the voice pipeline.
                pipeline = VoicePipeline(
                    workflow=SingleAgentVoiceWorkflow(agent),
                    config=VoicePipelineConfig(
                        tts_settings=TTSModelSettings(
                            voice=voice_name
                        ),
                        trace_include_sensitive_audio_data=False,
                    ),
                )

                # Run the voice pipeline and collect audio.
                async def process_voice():
                    result = await pipeline.run(audio_input)

                    audio_chunks = []

                    async for event in result.stream():
                        if event.type == "voice_stream_event_audio":
                            audio_chunks.append(
                                np.asarray(event.data).copy()
                            )

                    return result.total_output_text, audio_chunks

                with st.spinner(
                    "🎧 Listening to your question and preparing a response..."
                ):
                    answer, audio_chunks = asyncio.run(process_voice())

                st.session_state.answer = answer
                st.session_state.question = "Recorded voice question"

                if audio_chunks:
                    response_audio = np.concatenate(
                        audio_chunks,
                        axis=0,
                    )

                    st.session_state.audio_response = response_audio
                else:
                    st.session_state.audio_response = None

                st.session_state.history.append(
                    {
                        "question": st.session_state.question,
                        "answer": answer,
                    }
                )

                st.success("Your travel advice is ready!")

            except Exception as e:
                st.error(
                    f"Something went wrong: {e}"
                )

# --------------------------------------------------
# AI RESPONSE
# --------------------------------------------------

if st.session_state.answer:
    st.divider()

    st.markdown(
        '<div class="section-title">💬 Your AI Travel Advice</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="answer-card">
            {st.session_state.answer}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.audio_response is not None:
        st.markdown("### 🔊 Listen to the response")

        st.audio(
            st.session_state.audio_response,
            sample_rate=SAMPLE_RATE,
        )

    st.download_button(
        label="📄 Download Text Response",
        data=st.session_state.answer,
        file_name="travel_advisor_response.txt",
        mime="text/plain",
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.markdown(
    """
    <div style="text-align:center; padding:15px; color:#94a3b8;">
        ✈️ Travel AI Voice Advisor
        <br>
        <small>Powered by OpenAI Agents SDK and Streamlit</small>
    </div>
    """,
    unsafe_allow_html=True,
)
