# YouTube SEO Studio

Simple drag-and-drop app that takes a long video and generates:
- Summary
- Chapters
- SEO keywords
- Clickable titles + best title
- Modern thumbnail
- One `copy_paste_pack.txt` for fast upload

## Run locally
```bash
cd "/Users/haneymallemat/Documents/New project"
./run_ui.sh
```

## Deploy on Streamlit Cloud
1. Push this folder to GitHub repo `CriticalCareNow/youtube-seo-studio`
2. Go to https://share.streamlit.io/deploy
3. Choose repo + branch `main`
4. Main file path: `app.py`
5. Add secret:
```toml
OPENAI_API_KEY="your_key_here"
```
6. Deploy

## Important files
- `app.py`
- `src/youtube_seo_tool.py`
- `requirements.txt`
- `.streamlit/config.toml`
