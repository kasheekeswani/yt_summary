import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
import re

# Set Streamlit page config
st.set_page_config(page_title="YouTube Video Summarizer", layout="centered")

st.title("🎥 YouTube Video Summarizer")
st.write("Enter a YouTube URL to extract and summarize the transcript.")

# Input YouTube URL
youtube_url = st.text_input("🔗 Enter YouTube Video URL")

# Extract video ID from URL
def extract_video_id(url):
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    return match.group(1) if match else None

# Fetch transcript using youtube_transcript_api
def fetch_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join([entry['text'] for entry in transcript])
        return full_text
    except Exception as e:
        st.error(f"❌ Error fetching transcript: {e}")
        return None

# Summarize using Hugging Face pipeline
@st.cache_resource(show_spinner=False)
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text, max_chunk=1000):
    summarizer = load_summarizer()
    summaries = []
    for i in range(0, len(text), max_chunk):
        chunk = text[i:i+max_chunk]
        summary = summarizer(chunk, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
        summaries.append(summary)
    return " ".join(summaries)

# Run the app
if youtube_url:
    video_id = extract_video_id(youtube_url)
    if not video_id:
        st.warning("⚠️ Invalid YouTube URL")
    else:
        with st.spinner("⏳ Fetching transcript..."):
            transcript = fetch_transcript(video_id)

        if transcript:
            st.subheader("📜 Transcript")
            with st.expander("Click to view full transcript"):
                st.write(transcript)

            with st.spinner("🧠 Summarizing..."):
                summary = summarize_text(transcript)

            if summary:
                st.subheader("📘 Summary")
                st.success(summary)
                st.download_button("📥 Download Summary", summary, file_name="summary.txt")
