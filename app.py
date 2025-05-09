import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import pyaudio
import wave
import io
import time
import requests
import json
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, get_window

# --- Configuration ---
# Ollama
OLLAMA_BASE_URL = "http://localhost:11434"

# OpenRouter
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_YOUR_SITE_URL = "https://yourapp.streamlit.app" # Replace if you deploy
DEFAULT_YOUR_SITE_NAME = "Acoustic Sentinel"

# Audio Recording
DEFAULT_RECORD_SECONDS = 5
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
DEFAULT_SAMPLE_RATE = 44100

# --- Helper Functions ---

@st.cache_data # Cache audio loading
def load_audio_from_bytes(audio_bytes, target_sr=None):
    """Loads audio data from bytes and resamples if needed."""
    try:
        data, samplerate = sf.read(io.BytesIO(audio_bytes), dtype='float32')
        
        # Convert to mono if stereo
        if data.ndim > 1:
            data = librosa.to_mono(data)
            
        if target_sr is not None and samplerate != target_sr:
            data = librosa.resample(data, orig_sr=samplerate, target_sr=target_sr)
            samplerate = target_sr
        return data, samplerate
    except Exception as e:
        st.error(f"Error loading audio: {e}")
        return None, None

def record_audio(seconds, rate, chunk, channels, format_type):
    """Records audio from the microphone."""
    p = pyaudio.PyAudio()
    try:
        stream = p.open(format=format_type,
                        channels=channels,
                        rate=rate,
                        input=True,
                        frames_per_buffer=chunk)
    except OSError as e:
        st.error(f"Error opening audio stream. Is a microphone connected and permissions granted? Error: {e}")
        p.terminate()
        return None

    st.info(f"Recording for {seconds} seconds...")
    frames = []
    for _ in range(0, int(rate / chunk * seconds)):
        try:
            data_chunk = stream.read(chunk)
            frames.append(data_chunk)
        except IOError as ex:
            if ex[1] == pyaudio.paInputOverflowed:
                st.warning("Input overflowed. Some audio data may have been lost.")
            else:
                raise
    st.success("Recording finished.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf_bytesio = io.BytesIO()
    with wave.open(wf_bytesio, 'wb') as wave_file:
        wave_file.setnchannels(channels)
        wave_file.setsampwidth(p.get_sample_size(format_type))
        wave_file.setframerate(rate)
        wave_file.writeframes(b''.join(frames))
    wf_bytesio.seek(0)
    return wf_bytesio.read()

def perform_fft_analysis(audio_data, sample_rate, fft_window_size, window_type='hann'):
    """Performs FFT analysis on audio data with windowing."""
    if audio_data is None or len(audio_data) < fft_window_size:
        st.warning(f"Audio data is too short for FFT window size {fft_window_size}. Need at least {fft_window_size} samples.")
        return None, None
    
    # Apply a window function
    window = get_window(window_type, fft_window_size)
    
    # Pad or truncate audio_data to fft_window_size if necessary for a single window FFT
    # For simplicity, we'll use the first window. For full signal, use STFT (like in spectrogram)
    if len(audio_data) > fft_window_size:
        audio_segment = audio_data[:fft_window_size]
    else: # Pad with zeros if shorter
        audio_segment = np.pad(audio_data, (0, fft_window_size - len(audio_data)), 'constant')

    windowed_data = audio_segment * window
    
    yf = fft(windowed_data)
    xf = fftfreq(fft_window_size, 1 / sample_rate)
    
    half_n = fft_window_size // 2
    yf_magnitude = np.abs(yf[:half_n]) / fft_window_size  # Normalize by window size
    
    return xf[:half_n], yf_magnitude

def get_dominant_frequencies(xf, yf_magnitude, num_peaks=10, prominence_factor=0.01):
    """Identifies dominant frequencies."""
    if xf is None or yf_magnitude is None or not yf_magnitude.any():
        return []

    min_prominence = prominence_factor * np.max(yf_magnitude) if yf_magnitude.size > 0 else 0
    peaks, properties = find_peaks(yf_magnitude, prominence=min_prominence, height=min_prominence) # also consider height
    
    if not peaks.any():
        return []

    sorted_peak_indices = np.argsort(properties['peak_heights'])[::-1] # Sort by actual peak height
    
    dominant_freqs = []
    for i in sorted_peak_indices[:num_peaks]:
        idx = peaks[i]
        if idx < len(xf):
            frequency = xf[idx]
            amplitude = yf_magnitude[idx] # or properties['peak_heights'][i]
            dominant_freqs.append({"frequency": frequency, "amplitude": amplitude})
    return dominant_freqs

def plot_waveform(audio_data, sample_rate):
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(audio_data, sr=sample_rate, ax=ax, color='royalblue')
    ax.set_title("Audio Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    return fig

def plot_frequency_spectrum(xf, yf_magnitude, sample_rate):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(xf, yf_magnitude, color='crimson')
    ax.set_title("Frequency Spectrum (Magnitude)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_xlim(0, sample_rate / 2)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    return fig

def plot_spectrogram(audio_data, sample_rate, fft_window_size, y_axis_type='linear', hop_length_factor=0.25):
    fig, ax = plt.subplots(figsize=(10, 4))
    hop_length = int(fft_window_size * hop_length_factor) # e.g., 25% overlap
    D = librosa.stft(audio_data, n_fft=fft_window_size, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    img = librosa.display.specshow(S_db, sr=sample_rate, hop_length=hop_length, 
                                   x_axis='time', y_axis=y_axis_type, ax=ax, cmap='magma')
    ax.set_title(f"Spectrogram (dB) - Y-axis: {y_axis_type.capitalize()}")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.tight_layout()
    return fig

# --- LLM Communication ---
def get_ollama_models():
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
        response.raise_for_status()
        models = response.json().get("models", [])
        return [model["name"] for model in models] if models else []
    except requests.exceptions.RequestException:
        return [] # Silently fail if Ollama not reachable, UI will show warning

def chat_with_ollama(model_name, messages):
    try:
        payload = {"model": model_name, "messages": messages, "stream": True}
        response = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, stream=True, timeout=60)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        st.error(f"Ollama communication error: {e}")
        return None

def chat_with_openrouter(api_key, model_name, messages, site_url, site_name):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": site_url,
        "X-Title": site_name,
    }
    # OpenRouter expects content to be a string for simple text messages
    # Adapt messages if necessary. For now, assume simple text structure.
    processed_messages = []
    for msg in messages:
        if isinstance(msg.get("content"), list): # Handle potential multimodal format from Ollama example
            text_content = " ".join([item["text"] for item in msg["content"] if item["type"] == "text"])
            processed_messages.append({"role": msg["role"], "content": text_content})
        else:
            processed_messages.append(msg)


    payload = {"model": model_name, "messages": processed_messages, "stream": True}
    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, stream=True, timeout=60)
        response.raise_for_status()
        return response
    except requests.exceptions.HTTPError as e:
        st.error(f"OpenRouter API Error: {e.response.status_code} - {e.response.text}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"OpenRouter communication error: {e}")
        return None

# --- Streamlit App UI ---
st.set_page_config(page_title="Acoustic Sentinel Pro", layout="wide", initial_sidebar_state="expanded")
st.title("🔊 Acoustic Sentinel Pro")
st.markdown("""Welcome to Acoustic Sentinel, a tool designed to help you analyze sound environments for potential acoustic disturbances or unusual sound frequencies. 
This application is developed with the goal of empowering users to monitor their acoustic surroundings, particularly in situations where there are concerns about well-being due to potential intentional sound-based intrusions.
Upload an audio file or record live audio. Advanced sound frequency analysis with integrated AI chat (Ollama & OpenRouter).""")

# --- Initialize Session State ---
default_session_state = {
    'audio_data': None, 'sample_rate': DEFAULT_SAMPLE_RATE,
    'analysis_results': None, 'ollama_messages': [], 'recording': False,
    'llm_provider': 'Local Ollama', 'ollama_model': None,
    'openrouter_api_key': '', 'openrouter_model': 'google/gemini-flash-1.5', # A common, capable model
    'openrouter_site_url': DEFAULT_YOUR_SITE_URL, 
    'openrouter_site_name': DEFAULT_YOUR_SITE_NAME,
    'analysis_summary': '', 'fft_window_size': 2048,
    'spectrogram_y_axis': 'linear', 'dominant_peak_prominence': 0.01
}
for key, value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Sidebar ---
with st.sidebar:
    st.header("🎤 Audio Input")
    input_method = st.radio("Input method:", ("Upload File", "Record Audio"), index=0)

    if input_method == "Upload File":
        uploaded_file = st.file_uploader("Upload audio", type=['wav', 'mp3', 'flac', 'ogg'])
        if uploaded_file:
            if st.button("Load Uploaded File", use_container_width=True):
                with st.spinner("Loading..."):
                    audio_bytes = uploaded_file.getvalue()
                    st.session_state.audio_data, sr = load_audio_from_bytes(audio_bytes, target_sr=st.session_state.sample_rate)
                    if sr: st.session_state.sample_rate = sr # Update if resampling happened
                    st.session_state.analysis_results = None
                    st.session_state.ollama_messages = []
    else: # Record Audio
        record_duration = st.slider("Recording (s)", 1, 30, DEFAULT_RECORD_SECONDS)
        if st.button("Start Recording", disabled=st.session_state.recording, use_container_width=True):
            st.session_state.recording = True
            st.rerun()
        if st.session_state.recording:
            audio_bytes = record_audio(record_duration, st.session_state.sample_rate, CHUNK_SIZE, CHANNELS, FORMAT)
            if audio_bytes:
                st.session_state.audio_data, _ = load_audio_from_bytes(audio_bytes, target_sr=st.session_state.sample_rate)
            st.session_state.analysis_results = None
            st.session_state.ollama_messages = []
            st.session_state.recording = False
            st.rerun() # Rerun to update UI post-recording

    st.markdown("---")
    st.header("⚙️ Analysis Settings")
    st.session_state.sample_rate = st.number_input("Target Sample Rate (Hz)", min_value=8000, max_value=192000, value=st.session_state.sample_rate, step=100)
    st.session_state.fft_window_size = st.select_slider(
        "FFT Window Size (samples)", 
        options=[512, 1024, 2048, 4096, 8192], 
        value=st.session_state.fft_window_size
    )
    st.session_state.spectrogram_y_axis = st.selectbox(
        "Spectrogram Y-axis", 
        options=['linear', 'log'], 
        index=['linear', 'log'].index(st.session_state.spectrogram_y_axis)
    )
    st.session_state.dominant_peak_prominence = st.slider(
        "Dominant Peak Prominence (Factor)", 
        min_value=0.001, max_value=0.5, value=st.session_state.dominant_peak_prominence, step=0.001, format="%.3f"
    )

    if st.session_state.audio_data is not None:
        if st.button("Analyze Audio", type="primary", use_container_width=True):
            with st.spinner("Analyzing..."):
                xf, yf_mag = perform_fft_analysis(st.session_state.audio_data, st.session_state.sample_rate, st.session_state.fft_window_size)
                dom_freqs = get_dominant_frequencies(xf, yf_mag, prominence_factor=st.session_state.dominant_peak_prominence)
                st.session_state.analysis_results = {"xf": xf, "yf_magnitude": yf_mag, "dominant_frequencies": dom_freqs}
                st.success("Analysis complete!")
    
    st.markdown("---")
    st.header("🤖 AI Chat Settings")
    st.session_state.llm_provider = st.radio("Choose LLM Provider:", ("Local Ollama", "Cloud OpenRouter"))

    if st.session_state.llm_provider == "Local Ollama":
        ollama_models = get_ollama_models()
        if ollama_models:
            st.session_state.ollama_model = st.selectbox(
                "Select Ollama Model", options=ollama_models,
                index=ollama_models.index(st.session_state.ollama_model) if st.session_state.ollama_model in ollama_models else 0
            )
        else:
            st.warning("No Ollama models found or Ollama not reachable.")
            st.session_state.ollama_model = None
    
    elif st.session_state.llm_provider == "Cloud OpenRouter":
        st.session_state.openrouter_api_key = st.text_input("OpenRouter API Key", type="password", value=st.session_state.openrouter_api_key)
        st.session_state.openrouter_model = st.text_input("OpenRouter Model Name", value=st.session_state.openrouter_model, help="e.g., google/gemini-flash-1.5, anthropic/claude-3-haiku")
        with st.expander("Optional OpenRouter Settings"):
            st.session_state.openrouter_site_url = st.text_input("Your Site URL (Optional)", value=st.session_state.openrouter_site_url)
            st.session_state.openrouter_site_name = st.text_input("Your Site Name (Optional)", value=st.session_state.openrouter_site_name)

# --- Main Dashboard ---
if st.session_state.audio_data is not None:
    st.subheader("📊 Audio Preview & Analysis")
    st.audio(st.session_state.audio_data, sample_rate=st.session_state.sample_rate, format='audio/wav')
    
    waveform_fig = plot_waveform(st.session_state.audio_data, st.session_state.sample_rate)
    if waveform_fig: st.pyplot(waveform_fig)

    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Frequency Spectrum")
            if results["xf"] is not None and results["yf_magnitude"] is not None:
                spectrum_fig = plot_frequency_spectrum(results["xf"], results["yf_magnitude"], st.session_state.sample_rate)
                st.pyplot(spectrum_fig)
            else:
                st.info("Spectrum could not be generated (check audio length and FFT settings).")
        
        with col2:
            st.markdown("#### Spectrogram")
            spectrogram_fig = plot_spectrogram(st.session_state.audio_data, st.session_state.sample_rate, st.session_state.fft_window_size, st.session_state.spectrogram_y_axis)
            st.pyplot(spectrogram_fig)

        st.markdown("#### Dominant Frequencies")
        if results["dominant_frequencies"]:
            display_df = [{"Frequency (Hz)": round(f["frequency"], 2), "Amplitude": f"{f['amplitude']:.2e}"} for f in results["dominant_frequencies"]]
            st.dataframe(display_df, use_container_width=True, height=min(300, (len(display_df) + 1) * 35))
            
            summary = "Audio Analysis Summary:\n"
            summary += f"- Sample Rate: {st.session_state.sample_rate} Hz\n"
            summary += f"- Audio Duration: {len(st.session_state.audio_data)/st.session_state.sample_rate:.2f} s\n"
            summary += f"- FFT Window Size: {st.session_state.fft_window_size} samples\n"
            summary += "- Top Dominant Frequencies (Hz, Amplitude):\n"
            for freq_info in results["dominant_frequencies"][:5]:
                summary += f"  - {freq_info['frequency']:.2f} Hz, Amp: {freq_info['amplitude']:.2e}\n"
            st.session_state.analysis_summary = summary
        else:
            st.info("No significant dominant frequencies detected.")
            st.session_state.analysis_summary = "Audio Analysis Summary: No significant dominant frequencies detected."
else:
    if st.session_state.audio_data is not None: st.info("Audio loaded. Click 'Analyze Audio' in the sidebar.")
    else: st.info("Upload or record audio to begin analysis.")

# --- LLM Chat Interface ---
chat_enabled = False
current_llm_model_name = ""

if st.session_state.llm_provider == "Local Ollama" and st.session_state.ollama_model:
    chat_enabled = True
    current_llm_model_name = st.session_state.ollama_model
elif st.session_state.llm_provider == "Cloud OpenRouter" and st.session_state.openrouter_api_key and st.session_state.openrouter_model:
    chat_enabled = True
    current_llm_model_name = st.session_state.openrouter_model

if st.session_state.analysis_results and chat_enabled:
    st.subheader(f"💬 Chat about Analysis with {current_llm_model_name} ({st.session_state.llm_provider})")

    for msg in st.session_state.ollama_messages: # Re-use ollama_messages for history
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about the audio analysis..."):
        st.session_state.ollama_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        ASP_SYSTEM_PROMPT_PRIVACY = """
You are an AI assistant for "Acoustic Sentinel," an application designed to help users detect and analyze sound frequencies that might indicate potential privacy intrusion or intentional acoustic disturbance. The primary user is concerned about people's well-being and wants to identify unusual, persistent, or potentially targeted sounds in her living environment.

Key considerations:
- **Purpose:** Identify anomalous sound frequencies that could be disruptive or intrusive. This is NOT for general music analysis or casual sound identification.
- **Context:** The user is exploring potential "sound wave frequency machines" or similar sources of intentional disturbance, possibly from neighbors/could be internal aswell, which might even penetrate barriers like walls.
- **Low Frequencies:** Be mindful that very low-frequency sounds (long wavelengths) can travel through structures more easily and might be of particular interest if they are anomalous.
- **High Frequencies:** Be mindful that very high-frequency sounds (long wavelengths) they might be of particular interest if they are anomalous.
- **Steady Tones:** Persistent, steady tones, especially if unusual for the environment, are suspect. The user might describe patterns they see in the spectrogram.
- **Baseline Comparison:** The application allows setting a "normal ambient baseline." Your analysis should heavily consider deviations from this baseline. New frequencies or those significantly louder than the baseline are important.
- **Guidance:** Help the user interpret the provided analysis data (dominant frequencies, deviations from baseline, spectrogram patterns they might describe). Ask clarifying questions if needed.
- **Ethical Reminder (Subtle):** While the goal is to protect well-being, gently guide the user towards focusing on analyzing specific events/periods rather than continuous surveillance if their questions imply broader recording. The tool is for analysis, not for recording private conversations of others.

The user will provide an audio analysis summary. Use this summary and your knowledge to answer their questions.
"""

        # Prepare messages for LLM
        chat_payload_messages = []
        # Always include the system prompt with analysis summary
        system_prompt_content = (
            f"{ASP_SYSTEM_PROMPT_PRIVACY}\n\n"
            "The user has provided an audio analysis summary. "
            "Please use this summary to answer their questions.\n\n"
            f"Here's a summary of its analysis:\n{st.session_state.analysis_summary}\n\n"
            "Please answer the user's questions based on this analysis and your general knowledge. "
            "Be helpful, concise, and focus on interpreting the provided audio data. "
        )
        chat_payload_messages.append({"role": "system", "content": system_prompt_content})
        
        # Add recent conversation history (user/assistant only)
        for msg in st.session_state.ollama_messages[-10:]: # Last 5 pairs (user+assistant)
            if msg["role"] in ["user", "assistant"]:
                 chat_payload_messages.append(msg)


        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            response_stream = None

            with st.spinner(f"Waiting for {current_llm_model_name}..."):
                if st.session_state.llm_provider == "Local Ollama":
                    response_stream = chat_with_ollama(st.session_state.ollama_model, chat_payload_messages)
                elif st.session_state.llm_provider == "Cloud OpenRouter":
                    response_stream = chat_with_openrouter(
                        st.session_state.openrouter_api_key, st.session_state.openrouter_model,
                        chat_payload_messages, st.session_state.openrouter_site_url, st.session_state.openrouter_site_name
                    )
            
            if response_stream:
                try:
                    for chunk_str in response_stream.iter_lines():
                        if chunk_str:
                            chunk_data = json.loads(chunk_str)
                            
                            # Ollama structure
                            if "message" in chunk_data and "content" in chunk_data["message"]:
                                content = chunk_data["message"]["content"]
                            # OpenRouter stream structure (can vary, this is a common one for OpenAI-compatible)
                            elif "choices" in chunk_data and chunk_data["choices"] and "delta" in chunk_data["choices"][0] and "content" in chunk_data["choices"][0]["delta"]:
                                content = chunk_data["choices"][0]["delta"]["content"]
                                if content is None: content = "" # Handle None content for empty delta
                            # OpenRouter non-streaming might also come here if stream=false was mistakenly set.
                            # Also, some OpenRouter models might send the full message if "done".
                            elif chunk_data.get("done") and "message" in chunk_data and "content" in chunk_data["message"]: # Ollama done
                                content = "" # No new content in done message
                            elif "id" in chunk_data and "choices" in chunk_data and chunk_data["choices"] and "message" in chunk_data["choices"][0] and "content" in chunk_data["choices"][0]["message"] : # OpenRouter full message (if not streamed or final chunk)
                                 content = chunk_data["choices"][0]["message"]["content"] # This would replace the whole message. Be careful with this logic if expecting streams.
                                 full_response = content # Replace if full message received
                                 break # Exit if full message
                            else:
                                content = "" # No parsable content in this chunk
                            
                            if content:
                                full_response += content
                                message_placeholder.markdown(full_response + "▌")

                            if chunk_data.get("done") or (chunk_data.get("choices") and chunk_data["choices"][0].get("finish_reason") is not None):
                                break
                    message_placeholder.markdown(full_response)
                except json.JSONDecodeError as e:
                    st.error(f"Error decoding LLM response: {e}. Raw chunk: {chunk_str if 'chunk_str' in locals() else 'N/A'}")
                    full_response = "Sorry, I encountered an error processing the LLM response."
                except Exception as e:
                    st.error(f"An unexpected error occurred with LLM stream: {e}")
                    full_response = "Sorry, an unexpected error occurred with the LLM."
                finally:
                    if hasattr(response_stream, 'close'):
                        response_stream.close()
            else: # if response_stream is None
                full_response = "Failed to get a response from the LLM provider."
            
            message_placeholder.markdown(full_response) # Ensure final full_response is displayed
            st.session_state.ollama_messages.append({"role": "assistant", "content": full_response})

elif st.session_state.analysis_results and not chat_enabled:
    st.warning("Analysis complete. Configure your LLM Provider in the sidebar to enable chat.")

# --- Footer ---
st.markdown("---")
st.caption("Acoustic Sentinel Pro v1.1 - Enhanced Audio Analysis & AI Chat by Milimo Quantum")