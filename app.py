import streamlit as st
import whisper
from pydub import AudioSegment
import tempfile
import os

# Load the Whisper model
@st.cache_resource
def load_model():
    return whisper.load_model("base")  # You can use 'tiny', 'base', 'small', 'medium', or 'large'

model = load_model()

st.title("Whisper Transcription App")
st.subheader("Upload an audio file for transcription")

# File uploader
uploaded_file = st.file_uploader("Upload your audio file (mp3, wav, m4a)", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    # Play uploaded audio
    st.audio(uploaded_file, format="audio/mp3")

    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
        temp_audio_file.write(uploaded_file.read())
        temp_audio_file_path = temp_audio_file.name

    # Button to transcribe
    if st.button("Transcribe"):
        with st.spinner("Transcribing..."):
            try:
                # Convert audio to WAV format (if needed)
                audio = AudioSegment.from_file(temp_audio_file_path)
                wav_path = temp_audio_file_path.replace(".mp3", ".wav")
                audio.export(wav_path, format="wav")

                # Perform transcription
                result = model.transcribe(wav_path)
                transcription = result["text"]

                # Display transcription
                st.success("Transcription Completed!")
                st.text_area("Transcription", transcription, height=200)

                # Add download button for transcription
                st.download_button(
                    label="Download Transcription",
                    data=transcription,
                    file_name="transcription.txt",
                    mime="text/plain"
                )

                # Cleanup temporary files
                os.remove(wav_path)

            except Exception as e:
                st.error(f"Error during transcription: {e}")

    # Cleanup uploaded file after transcription
    os.remove(temp_audio_file_path)
