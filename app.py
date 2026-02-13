#!/usr/bin/env python3
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st

from src.youtube_seo_tool import run_pipeline

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_DIR / 'output' / 'app-runs'

st.set_page_config(page_title='YouTube SEO Studio', page_icon='🎬', layout='wide')
st.title('YouTube SEO Studio')
st.caption('Drop your video and optional speaker image. Copy/paste-ready output in one click.')

with st.sidebar:
    st.header('Settings')
    api_key = st.text_input('OpenAI API Key', type='password', value='')
    model = st.text_input('SEO Model', value='gpt-4.1')
    transcript_model = st.text_input('Transcription Model', value='gpt-4o-mini-transcribe')
    chapter_style = st.selectbox('Chapter Style', ['super_short', 'short', 'standard'], index=0)
    thumbnail_style = st.selectbox('Thumbnail Style', ['ref_style_3', 'viral_2026', 'clean_2026'], index=0)
    top_n_titles = st.slider('Title options', min_value=5, max_value=20, value=10)

left, right = st.columns([2, 1])
with left:
    video_upload = st.file_uploader('Video File', type=['mp4', 'mov', 'mkv', 'm4v', 'webm'])
    episode_title = st.text_input('Episode Title', value='')
    creator_name = st.text_input('Creator Name', value='Reuben Strayer')
with right:
    speaker_upload = st.file_uploader('Speaker Image (optional)', type=['png', 'jpg', 'jpeg', 'webp'])

if st.button('Generate Package', type='primary', use_container_width=True):
    if not api_key.strip():
        st.error('Add your OpenAI API key.')
        st.stop()
    if not video_upload:
        st.error('Upload a video file.')
        st.stop()

    run_id = datetime.now().strftime('%Y%m%d-%H%M%S')
    output_dir = OUTPUT_ROOT / f'run-{run_id}'
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix='ytstudio_') as tmpdir:
        tmp = Path(tmpdir)
        video_path = tmp / video_upload.name
        video_path.write_bytes(video_upload.getbuffer())

        speaker_path = None
        if speaker_upload:
            speaker_path = tmp / speaker_upload.name
            speaker_path.write_bytes(speaker_upload.getbuffer())

        with st.spinner('Processing...'):
            result = run_pipeline(
                url=None,
                video_file=video_path,
                transcript_file=None,
                episode_title=episode_title.strip() or None,
                creator_name=creator_name.strip() or None,
                output_dir=output_dir,
                openai_api_key=api_key.strip(),
                model=model,
                transcript_model=transcript_model,
                top_n_titles=top_n_titles,
                chapter_style=chapter_style,
                speaker_asset=speaker_path,
                thumbnail_style=thumbnail_style,
            )

    st.success(f"Done. {result['output_dir']}")

    pack_text = (output_dir / 'copy_paste_pack.txt').read_text(encoding='utf-8')
    chapters_text = (output_dir / 'chapters_youtube.txt').read_text(encoding='utf-8')
    tags_text = (output_dir / 'keywords_comma.txt').read_text(encoding='utf-8')
    thumb_path = output_dir / 'thumbnail.png'

    tabs = st.tabs(['Copy/Paste Pack', 'Chapters', 'Tags', 'Thumbnail', 'JSON'])

    with tabs[0]:
        st.text_area('Everything', value=pack_text, height=500)
        st.download_button('Download copy_paste_pack.txt', data=pack_text, file_name='copy_paste_pack.txt')

    with tabs[1]:
        st.text_area('YouTube Chapters', value=chapters_text, height=330)
        st.download_button('Download chapters_youtube.txt', data=chapters_text, file_name='chapters_youtube.txt')

    with tabs[2]:
        st.text_area('Tags', value=tags_text, height=180)
        st.download_button('Download keywords_comma.txt', data=tags_text, file_name='keywords_comma.txt')

    with tabs[3]:
        st.image(str(thumb_path), caption='Generated Thumbnail', use_container_width=True)
        st.download_button('Download thumbnail.png', data=thumb_path.read_bytes(), file_name='thumbnail.png')

    with tabs[4]:
        st.code((output_dir / 'seo_package.json').read_text(encoding='utf-8'), language='json')
