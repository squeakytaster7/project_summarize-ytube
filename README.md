# 🎬 YouTube Video Summarizer (Indonesia + English)

Aplikasi ini secara otomatis mengambil teks dari video YouTube (via transcript atau transkripsi audio), lalu merangkumnya menjadi satu paragraf atau poin-poin penting.  
Dibangun menggunakan Python, Streamlit, Whisper, dan Transformers (mT5).

---

## 🚀 Fitur Utama

- 🔍 **Ambil transcript otomatis** (bahasa Indonesia & Inggris)
- 🎧 **Fallback transkripsi audio** menggunakan OpenAI Whisper jika transcript tidak tersedia
- 🧠 **Ringkasan otomatis** dengan model mT5 (multibahasa)
- 🧩 **Interface Streamlit interaktif**
- 💾 Bisa dijalankan langsung dari **Jupyter Notebook** untuk eksperimen awal

---

## 🧰 Teknologi yang Digunakan

| Komponen | Deskripsi |
|-----------|------------|
| `youtube-transcript-api` | Mengambil teks (subtitle) dari video YouTube |
| `yt-dlp` | Mengunduh audio dari YouTube (untuk Whisper) |
| `openai-whisper` | Speech-to-text lokal, mendukung bahasa Indonesia |
| `transformers` + `sentencepiece` | Model ringkasan mT5 multilingual |
| `streamlit` | Antarmuka web sederhana dan interaktif |
| `torch` | Backend ML untuk model-model di atas |
