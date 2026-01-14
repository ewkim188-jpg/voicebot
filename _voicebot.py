# voicebot.py
# 실행: python -m streamlit run voicebot.py
# 필요 패키지:
#   python -m pip install streamlit openai gtts audio-recorder-streamlit
# (gTTS 관련 click 충돌이 나면: python -m pip install "click<8.2")

import io
import os
from datetime import datetime

import streamlit as st
from openai import OpenAI
from audio_recorder_streamlit import audio_recorder
from gtts import gTTS


# -----------------------------
# STT: WAV bytes -> text
# -----------------------------
def STT(wav_bytes: bytes, apikey: str) -> str:
    """
    audio_recorder_streamlit.audio_recorder()는 보통
    - 녹음 전: None
    - 녹음 후: wav bytes
    를 반환합니다. bytes를 BytesIO로 감싸서 OpenAI Transcribe에 전달합니다.
    """
    client = OpenAI(api_key=apikey)

    f = io.BytesIO(wav_bytes)
    f.name = "input.wav"  # API가 파일명을 필요로 하는 경우가 있어 지정

    # 계정/SDK 환경에 따라 whisper-1도 사용 가능
    r = client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",
        file=f,
    )
    return r.text


# -----------------------------
# GPT: messages -> answer
# -----------------------------
def ask_gpt(messages: list[dict], model: str, apikey: str) -> str:
    """
    최신 SDK에서 안정적으로 동작하도록 responses API를 사용합니다.
    (messages 형식을 그대로 ChatCompletions로 보내는 방식은 구/신 혼용으로 자주 깨짐)
    """
    client = OpenAI(api_key=apikey)

    # messages를 단순 텍스트로 합쳐 input으로 전달(가장 튼튼한 방식)
    prompt = "\n".join([f'{m["role"]}: {m["content"]}' for m in messages])

    resp = client.responses.create(
        model=model,
        input=prompt,
    )
    return resp.output_text


# -----------------------------
# TTS: text -> play mp3
# -----------------------------
def TTS_play(text: str):
    filename = "output.mp3"
    gTTS(text=text, lang="ko").save(filename)

    with open(filename, "rb") as f:
        data = f.read()

    try:
        os.remove(filename)
    except OSError:
        pass

    st.audio(data, format="audio/mp3")


# -----------------------------
# Main App
# -----------------------------
def main():
    st.set_page_config(page_title="음성 비서 프로그램", layout="wide")
    st.header("음성 비서 프로그램")
    st.markdown("---")

    # ---- session_state init ----
    st.session_state.setdefault("chat", [])  # [("user"/"bot", "HH:MM", "text")]
    st.session_state.setdefault("OPENAI_API", "")
    st.session_state.setdefault(
        "messages",
        [{"role": "system", "content": "You are a thoughtful assistant. Answer in Korean. Keep it concise."}],
    )
    st.session_state.setdefault("check_reset", False)
    st.session_state.setdefault("last_answer", "")

    # ---- info ----
    with st.expander("음성비서 프로그램에 관하여", expanded=True):
        st.write(
            """
        • 음성 비서 프로그램의 UI는 **스트림릿(Streamlit)** 을 활용했습니다.  

        • STT(Speech-To-Text)는 **OpenAI의 Whisper AI 모델**을 활용하여  
          사용자의 음성을 텍스트로 변환합니다.  

        • 변환된 텍스트에 대한 답변은 **OpenAI의 GPT 모델**을 활용하여  
          자연스러운 대화를 생성합니다.  

        • TTS(Text-To-Speech)는 **구글의 Google Translate TTS(gTTS)** 를 활용하여  
          생성된 답변을 음성으로 출력합니다.
        """
        )

    # ---- sidebar ----
    with st.sidebar:
        st.session_state["OPENAI_API"] = st.text_input(
            "OPENAI API 키", placeholder="Enter your API key", value="", type="password"
        )

        st.markdown("---")

        # 모델은 실제로 존재하는 것을 쓰는 편이 안전합니다.
        # (원하면 gpt-4 / gpt-3.5-turbo 그대로 두셔도 되지만 계정/SDK에 따라 에러가 날 수 있어
        #  gpt-4o-mini 같은 최신 모델을 권장)
        model = st.radio("GPT 모델", options=["gpt-4o-mini", "gpt-4.1-mini", "gpt-3.5-turbo"])

        st.markdown("---")

        if st.button("초기화"):
            st.session_state["chat"] = []
            st.session_state["messages"] = [
                {"role": "system", "content": "You are a thoughtful assistant. Answer in Korean. Keep it concise."}
            ]
            st.session_state["check_reset"] = False
            st.session_state["last_answer"] = ""
            st.rerun()

    # ---- layout ----
    col1, col2 = st.columns(2)

    # -----------------------------
    # Left: Record & Ask
    # -----------------------------
    with col1:
        st.subheader("질문하기")
        st.markdown("### 🎤 클릭하여 녹음하기")
        st.markdown(
            """
            <style>
            /* audio-recorder 내부 텍스트 숨기기 */
            div[data-testid="stAudioRecorder"] span {
            display: none;
            }

            /* 마이크 아이콘 숨기기 */
            div[data-testid="stAudioRecorder"] svg {
            display: none;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        # 녹음 전: None / 녹음 후: wav bytes
        audio_bytes = audio_recorder()

        # 녹음된 데이터가 있을 때만 처리 (None 방지)
        if audio_bytes and (st.session_state["check_reset"] is False):
            st.audio(audio_bytes, format="audio/wav")
            if not audio_bytes:
                st.warning("먼저 녹음한 뒤 버튼을 눌러주세요.")
            if st.button("질문 보내기"):
                if not st.session_state["OPENAI_API"]:
                    st.warning("API 키를 먼저 입력하세요.")
                else:
                    # STT
                    question = STT(audio_bytes, st.session_state["OPENAI_API"])

                    now = datetime.now().strftime("%H:%M")
                    st.session_state["chat"].append(("user", now, question))
                    st.session_state["messages"].append({"role": "user", "content": question})

                    # GPT
                    answer = ask_gpt(st.session_state["messages"], model, st.session_state["OPENAI_API"])

                    now = datetime.now().strftime("%H:%M")
                    st.session_state["chat"].append(("bot", now, answer))
                    st.session_state["messages"].append({"role": "assistant", "content": answer})

                    st.session_state["last_answer"] = answer

                    # Streamlit은 스크립트를 재실행하므로, 결과가 즉시 오른쪽에 보이게 rerun
                    st.rerun()

    # -----------------------------
    # Right: Chat & Speak Answer
    # -----------------------------
    with col2:
        st.subheader("질문/답변")

        if not st.session_state["chat"]:
            st.info("왼쪽에서 녹음 후 '질문 보내기'를 눌러보세요.")
        else:
            # 간단 텍스트 로그 형태
            for sender, t, msg in st.session_state["chat"]:
                st.write(f"[{t}] {sender}: {msg}")

        if st.session_state["last_answer"]:
            st.markdown("---")
            if st.button("답변 음성으로 듣기"):
                TTS_play(st.session_state["last_answer"])


main()
