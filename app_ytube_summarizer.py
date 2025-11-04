import re
import tempfile
from pathlib import Path


import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript
import yt_dlp
import whisper

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------- Utils ----------
def extract_video_id(url: str) -> str:
    # Support youtu.be/VIDEO_ID and youtube.com/watch?v=VIDEO_ID
    m = re.search(r'(?:v=|youtu\.be/)([A-Za-z0-9_\-]{11})', url)
    if not m:
        raise ValueError("URL YouTube tidak valid atau VIDEO_ID tidak ditemukan.")
    return m.group(1)

def fetch_transcript(video_id: str, languages=("id","en")) -> str | None:
    try:
        # Prioritaskan ID, lalu EN, lalu auto-generated
        tr_list = YouTubeTranscriptApi.list_transcripts(video_id)
        candidates = []
        for lang in languages:
            if tr_list.find_manually_created_transcript([lang]):
                candidates.append(tr_list.find_manually_created_transcript([lang]))
            if tr_list.find_generated_transcript([lang]):
                candidates.append(tr_list.find_generated_transcript([lang]))
        if not candidates:
            # jika tak ada yang cocok, coba apapun yang ada
            for tr in tr_list:
                candidates.append(tr)

        for tr in candidates:
            try:
                data = tr.fetch()
                text = " ".join([x["text"] for x in data if x["text"].strip()])
                return text
            except:
                continue
        return None
    except (TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript):
        return None

def download_audio(url: str, outdir: Path) -> Path:
    outtmpl = str(outdir / "%(id)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "128",
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        vid = info.get("id")
        audio_path = outdir / f"{vid}.mp3"
        if not audio_path.exists():
            # fallback kalau postprocessor beda ekstensi
            for p in outdir.glob(f"{vid}.*"):
                if p.suffix.lower() in [".m4a", ".webm", ".mp3", ".opus"]:
                    return p
        return audio_path

def transcribe_audio(audio_path: Path, model_name="small") -> str:
    # model: tiny/base/small/medium/large
    model = whisper.load_model(model_name)
    result = model.transcribe(str(audio_path))
    return result.get("text", "").strip()

def chunk_text(text: str, max_chars=1800, overlap=200):
    text = re.sub(r'\s+', ' ', text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        # coba putus di titik terdekat biar rapi
        period = text.rfind(". ", start, end)
        if period == -1 or period <= start + 300:
            period = end
        else:
            period += 1
        chunks.append(text[start:period].strip())
        start = max(0, period - overlap)
    return [c for c in chunks if c]

@st.cache_resource
def load_summarizer():
    model_id = "csebuetnlp/mT5_multilingual_XLSum"
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    return tok, mdl

def summarize_chunks(chunks, target_lang="id", max_len=120):
    tok, mdl = load_summarizer()

    def _summ(txt):
        # mT5 XLSum biasanya pakai prefix untuk ringkasan
        prompt = f"summarize: {txt}"
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1024)
        outputs = mdl.generate(
            **inputs,
            max_length=max_len,
            num_beams=4,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
        return tok.decode(outputs[0], skip_special_tokens=True)

    # Map: ringkas per chunk
    partial = [_summ(c) for c in chunks]
    # Reduce: gabungkan lalu ringkas lagi jadi 1 paragraf
    combined = " ".join(partial)
    final = _summ(combined)
    return final, partial

# ---------- Streamlit UI ----------
st.set_page_config(page_title="YouTube → Summary", page_icon="📝", layout="centered")
st.title("📝 Ringkas Video YouTube")

url = st.text_input("Tempel URL YouTube di sini")
mode = st.radio("Cara ambil teks:", ["Auto (Transcript → Whisper fallback)", "Selalu Transcript", "Selalu Whisper"], index=0)
whisper_model = st.select_slider("Whisper model (kalau dipakai)", options=["tiny","base","small","medium"], value="small")
run = st.button("Ringkas sekarang")

if run:
    if not url.strip():
        st.error("Mohon masukkan URL YouTube.")
        st.stop()

    with st.spinner("Memproses…"):
        try:
            vid = extract_video_id(url)
        except Exception as e:
            st.error(f"URL tidak valid: {e}")
            st.stop()

        transcript_text = None
        if mode in ("Auto (Transcript → Whisper fallback)", "Selalu Transcript"):
            transcript_text = fetch_transcript(vid)

        if transcript_text:
            st.success("Berhasil mengambil transcript dari YouTube.")
            source_text = transcript_text
        else:
            if mode == "Selalu Transcript":
                st.error("Transcript tidak tersedia untuk video ini.")
                st.stop()
            # Fallback ke Whisper
            try:
                with tempfile.TemporaryDirectory() as td:
                    td = Path(td)
                    audio = download_audio(url, td)
                    st.info("Transcript tidak ada. Menggunakan Whisper untuk transkripsi audio…")
                    source_text = transcribe_audio(audio, model_name=whisper_model)
            except Exception as e:
                st.error(f"Gagal transcribe audio: {e}")
                st.stop()

        if not source_text or len(source_text) < 50:
            st.error("Teks terlalu pendek atau kosong setelah ekstraksi/transkripsi.")
            st.stop()

        chunks = chunk_text(source_text)
        summary, bullets = summarize_chunks(chunks, target_lang="id", max_len=160)

        st.subheader("Ringkasan (satu paragraf)")
        st.write(summary)

        with st.expander("Lihat ringkasan per-bagian"):
            for i, b in enumerate(bullets, 1):
                st.markdown(f"**Bagian {i}**: {b}")

        with st.expander("Lihat transcript sumber (bersih)"):
            st.write(source_text[:12000] + ("..." if len(source_text) > 12000 else ""))
