# Acoustic Sentinel Pro 🔊

**Version:** 1.1
**Description:** An advanced audio analysis application built with Streamlit, enabling users to record or upload audio, perform detailed frequency analysis, and discuss the results with AI language models (supporting local Ollama and cloud-based OpenRouter).

## Features

*   **Flexible Audio Input:**
    *   Upload common audio file formats (`.wav`, `.mp3`, `.flac`, `.ogg`).
    *   Record audio directly from your microphone.
*   **Comprehensive Audio Analysis:**
    *   **Waveform Visualization:** See the raw audio signal over time.
    *   **Frequency Spectrum:** View the magnitude of different frequencies present in the audio using Fast Fourier Transform (FFT).
    *   **Spectrogram:** Analyze how the frequency content of the audio changes over time.
    *   **Dominant Frequency Detection:** Identifies and lists the most prominent frequencies and their amplitudes.
*   **Configurable Analysis Parameters:**
    *   Adjust the target **Sample Rate** for processing.
    *   Select the **FFT Window Size** for frequency resolution.
    *   Choose **Spectrogram Y-axis** scaling (linear Hz, log).
    *   Fine-tune **Dominant Peak Prominence** for sensitivity in frequency detection.
*   **Integrated AI Chat for Interpretation:**
    *   **Dual LLM Provider Support:**
        *   **Local Ollama:** Utilize models running on your local Ollama instance.
        *   **Cloud OpenRouter:** Connect to a wide range of models via OpenRouter (requires API key).
    *   **Contextual Discussion:** The AI is provided with a summary of the audio analysis to enable informed discussions about the findings.
*   **User-Friendly Interface:**
    *   Clean sidebar for controls and settings.
    *   Interactive dashboard for visualizing analysis results.
    *   Real-time chat interface.

## Use Cases

*   **Identifying unusual or persistent sounds:** Detect specific frequencies that might indicate machinery noise, electronic hums, or other potential disturbances.
*   **Acoustic environment monitoring:** Get a general understanding of the sound profile of an environment.
*   **Educational tool:** Learn about audio signals, frequencies, and spectrum analysis.
*   **Troubleshooting audio issues:** Help pinpoint feedback loops or unwanted resonances.
*   **Privacy-conscious sound awareness:** Analyze sounds for potential intrusive frequencies without necessarily recording full conversations (though users should always be mindful of legal and ethical considerations).

## Prerequisites

1.  **Python:** Version 3.8 or higher.
2.  **Pip:** Python package installer.
3.  **PortAudio:** Required by `pyaudio` for microphone access.
    *   **macOS:** `brew install portaudio`
    *   **Debian/Ubuntu:** `sudo apt-get install portaudio19-dev python3-pyaudio`
    *   **Windows:** Download pre-compiled `pyaudio` wheels or install via Conda, ensuring PortAudio is available.
4.  **(Optional) Ollama:** If using local LLMs.
    *   Install Ollama from [ollama.ai](https://ollama.ai/).
    *   Pull desired models (e.g., `ollama pull llama3`).
5.  **(Optional) OpenRouter Account:** If using cloud-based LLMs.
    *   Create an account at [openrouter.ai](https://openrouter.ai/).
    *   Obtain an API Key.

## Installation

1.  **Clone the repository or download the `acoustic_sentinel_enhanced.py` file.**
2.  **Navigate to the project directory in your terminal.**
3.  **Install required Python libraries:**
    ```bash
    pip install streamlit pyaudio soundfile librosa scipy numpy matplotlib requests
    ```

## How to Run

1.  **Ensure Ollama is running** if you plan to use local models.
2.  **Open your terminal and navigate to the directory where you saved `acoustic_sentinel_enhanced.py`.**
3.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
4.  **The application will open in your default web browser.**

## Usage Guide

1.  **Audio Input (Sidebar):**
    *   Choose "Upload File" to select an audio file from your computer. Click "Load Uploaded File."
    *   Choose "Record Audio" to record using your microphone. Adjust the duration and click "Start Recording."
2.  **Analysis Settings (Sidebar):**
    *   Adjust parameters like Sample Rate, FFT Window Size, Spectrogram Y-axis, and Dominant Peak Prominence as needed.
3.  **Analyze Audio:**
    *   Once audio is loaded or recorded, click the "Analyze Audio" button in the sidebar.
4.  **View Results (Main Dashboard):**
    *   The waveform, frequency spectrum, spectrogram, and a table of dominant frequencies will be displayed.
5.  **AI Chat (Sidebar & Main Dashboard):**
    *   **Configure LLM Provider:** In the sidebar, select "Local Ollama" or "Cloud OpenRouter."
        *   **Ollama:** Select an available model from the dropdown.
        *   **OpenRouter:** Enter your API Key and the desired model name (e.g., `google/gemini-flash-1.5`). Optionally, provide your Site URL and Name for OpenRouter rankings.
    *   **Chat:** Once analysis is complete and an LLM is configured, a chat interface will appear below the analysis results. Type your questions about the audio analysis.

## Future Enhancements

*   **Real-time/Chunked Analysis:** For live "sentinel" capabilities.
*   **Baseline Anomaly Detection:** Compare current audio to a "normal" baseline.
*   **Event Logging & Alerts:** For significant sound events.
*   **Interactive Plots:** Using libraries like Plotly.
*   **Export Functionality:** Save analysis reports or plots.
*   **Advanced Audio Features:** MFCCs, ZCR, etc., for broader sound characterization.

## Important Considerations

*   **Microphone Permissions:** Ensure your browser and operating system have granted microphone access to the application if you intend to record audio.
*   **Computational Load:** Analyzing very long audio files or using very small FFT window sizes can be computationally intensive.
*   **Ollama Performance:** The speed of local Ollama models depends on your hardware.
*   **OpenRouter Costs:** Be mindful of API usage costs if using OpenRouter extensively.
*   **Privacy and Ethics:** Always use this tool responsibly and be aware of legal and ethical implications when recording or analyzing audio, especially in shared or private spaces. This tool is designed for frequency analysis and not for transcribing speech or eavesdropping.

## Contributing

Contributions, bug reports, and feature requests are welcome! Please open an issue or submit a pull request.

## License

This project is open-source. Please specify a license if you intend to distribute it widely (e.g., MIT License).