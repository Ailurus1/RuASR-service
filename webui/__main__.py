import os
from typing import Optional

import gradio as gr
import httpx


ASR_URL = os.environ.get("ASR_URL", "http://localhost:9090/asr/")
TIMEOUT = float(os.environ.get("ASR_TIMEOUT", "120"))


def transcribe(audio_file: Optional[str]) -> str:
    if not audio_file:
        return "Please record or upload an audio file."

    try:
        with open(audio_file, "rb") as f:
            files = {"audio_message": ("audio_message.wav", f.read())}

        with httpx.Client(timeout=TIMEOUT) as client:
            resp = client.post(ASR_URL, files=files)

        resp.raise_for_status()
        data = resp.json()

        if "transcription" in data:
            return data["transcription"]

        return f"ASR service error: {data.get('error', 'Unknown error')}"
    except Exception as e:
        return f"Request failed: {e}"


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="RuASR Web UI") as demo:
        gr.Markdown(
            """
            # RuASR Web UI

            Record or upload an audio file and get a transcription
            from the RuASR inference service.
            """
        )

        with gr.Row():
            audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Audio")
        output = gr.Textbox(label="Transcription", lines=8)

        submit = gr.Button("Transcribe")
        submit.click(fn=transcribe, inputs=[audio], outputs=[output])

    return demo


def main() -> None:
    demo = build_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("WEBUI_PORT", "7860")),
    )


if __name__ == "__main__":
    main()

