#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from openai import OpenAI
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from youtube_transcript_api import YouTubeTranscriptApi
from yt_dlp import YoutubeDL


@dataclass
class VideoContext:
    video_id: str
    source_ref: str
    title: str
    creator: str
    duration_seconds: Optional[int]


def extract_video_id(url: str) -> str:
    parsed = urlparse(url)

    if parsed.hostname in {"youtu.be"}:
        return parsed.path.strip("/")

    if parsed.hostname and "youtube.com" in parsed.hostname:
        if parsed.path == "/watch":
            ids = parse_qs(parsed.query).get("v")
            if ids:
                return ids[0]
        if parsed.path.startswith("/shorts/"):
            return parsed.path.split("/shorts/")[-1].split("/")[0]

    raise ValueError(f"Could not extract a YouTube video ID from URL: {url}")


def fetch_video_metadata(url: str) -> Dict:
    with YoutubeDL({"quiet": True, "skip_download": True, "noplaylist": True}) as ydl:
        info = ydl.extract_info(url, download=False)
    return {
        "title": info.get("title") or "Untitled Episode",
        "creator": info.get("uploader") or "Unknown Creator",
        "duration": info.get("duration"),
    }


def load_transcript_from_youtube(video_id: str) -> List[Dict]:
    try:
        return YouTubeTranscriptApi.get_transcript(
            video_id,
            languages=["en", "en-US", "en-GB"],
            preserve_formatting=False,
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not fetch YouTube transcript. Use --transcript-file or --video-file mode."
        ) from exc


def load_transcript_file(transcript_file: Path) -> List[Dict]:
    with transcript_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Transcript file must be a JSON list of {start, duration, text} objects.")
    return data


def transcribe_video_file(client: OpenAI, video_file: Path, transcript_model: str) -> List[Dict]:
    if not video_file.exists():
        raise FileNotFoundError(f"Video file not found: {video_file}")

    with tempfile.TemporaryDirectory(prefix="ytseo_") as tmpdir:
        tmp = Path(tmpdir)
        chunk_pattern = tmp / "chunk_%03d.mp3"

        ffmpeg_bin = "ffmpeg"
        try:
            import imageio_ffmpeg

            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass

        cmd = [
            ffmpeg_bin,
            "-y",
            "-i",
            str(video_file),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-b:a",
            "24k",
            "-f",
            "segment",
            "-segment_time",
            "600",
            "-reset_timestamps",
            "1",
            str(chunk_pattern),
        ]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg not found. Install requirements (imageio-ffmpeg).") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError("ffmpeg failed while processing video.") from exc

        chunks = sorted(tmp.glob("chunk_*.mp3"))
        if not chunks:
            raise RuntimeError("No audio chunks created from video.")

        segments: List[Dict] = []
        for idx, chunk in enumerate(chunks):
            offset = float(idx * 600)
            with chunk.open("rb") as audio:
                resp = client.audio.transcriptions.create(
                    model=transcript_model,
                    file=audio,
                    response_format="json",
                )

            text = (getattr(resp, "text", "") or "").strip()
            if text:
                segments.append({"start": offset, "duration": 600.0, "text": text})

        if not segments:
            raise RuntimeError("Transcription returned no text.")

        return segments


def seconds_to_timestamp(seconds: float) -> str:
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def compress_transcript(segments: List[Dict], bucket_seconds: int = 90, max_chars: int = 120000) -> str:
    buckets: Dict[int, List[str]] = {}

    for seg in segments:
        start = float(seg.get("start", 0))
        text = re.sub(r"\s+", " ", str(seg.get("text", "")).strip())
        if not text:
            continue
        bucket = int(start // bucket_seconds)
        buckets.setdefault(bucket, []).append(text)

    lines: List[str] = []
    for bucket in sorted(buckets.keys()):
        ts = seconds_to_timestamp(bucket * bucket_seconds)
        merged = " ".join(buckets[bucket]).strip()
        lines.append(f"{ts} | {merged}")

    transcript = "\n".join(lines)
    return transcript[:max_chars]


def generate_seo_package(
    client: OpenAI,
    model: str,
    context: VideoContext,
    transcript_text: str,
    top_n_titles: int,
) -> Dict:
    prompt = f"""
Build a YouTube growth package from this long-form episode.

Video metadata:
- Title: {context.title}
- Creator: {context.creator}
- Duration seconds: {context.duration_seconds}
- Source: {context.source_ref}

Transcript (timestamped):
{transcript_text}

Return strict JSON only with this schema:
{{
  "high_impact_summary": "2-4 paragraphs",
  "chapter_suggestions": [
    {{"timestamp":"MM:SS or HH:MM:SS","title":"short title","what_happens":"1 sentence","seo_hook":"1 sentence"}}
  ],
  "seo_keywords": ["keyword"],
  "title_options": ["title"],
  "best_title": "best title",
  "best_title_reason": "1-2 sentences",
  "thumbnail_copy": {{
    "primary_text": "2-6 words",
    "secondary_text": "2-5 words",
    "visual_hook": "short visual direction"
  }}
}}

Rules:
- 12-20 chapters.
- Chapter titles must be concise.
- 25-45 searchable keywords.
- Generate exactly {top_n_titles} title options, strongest first.
- Keep language compelling but not misleading.
"""

    completion = client.chat.completions.create(
        model=model,
        temperature=0.7,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are an expert YouTube growth strategist. Return JSON only."},
            {"role": "user", "content": prompt},
        ],
    )

    content = completion.choices[0].message.content
    if not content:
        raise RuntimeError("Model returned empty content.")

    return json.loads(content)


def normalize_text_for_thumb(text: str, fallback: str, max_words: int) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if not cleaned:
        cleaned = fallback
    words = cleaned.split()
    return " ".join(words[:max_words])


def shorten_chapter_title(title: str, style: str = "super_short") -> str:
    style_map = {
        "standard": (10, 64),
        "short": (6, 42),
        "super_short": (4, 30),
    }
    max_words, max_chars = style_map.get(style, (4, 30))

    cleaned = re.sub(r"\s+", " ", str(title or "")).strip()
    if not cleaned:
        return "Key Point"

    words = cleaned.split()[:max_words]
    tokens: List[str] = []
    for word in words:
        candidate = " ".join(tokens + [word])
        if len(candidate) <= max_chars:
            tokens.append(word)
        else:
            break

    if not tokens:
        tokens = [words[0][:max_chars]]

    dangling = {"and", "or", "vs", "with", "to", "of", "the", "a", "an", "+", "&"}
    while len(tokens) > 1 and tokens[-1].lower().strip(".,:;!?") in dangling:
        tokens.pop()

    return " ".join(tokens)


def normalize_chapters(chapters: List[Dict], chapter_style: str) -> List[Dict]:
    normalized: List[Dict] = []
    for row in chapters:
        normalized.append(
            {
                "timestamp": str(row.get("timestamp", "")).strip(),
                "title": shorten_chapter_title(str(row.get("title", "")), style=chapter_style),
                "what_happens": str(row.get("what_happens", "")).strip(),
                "seo_hook": str(row.get("seo_hook", "")).strip(),
            }
        )
    return normalized


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_copy_paste_pack(output_dir: Path, data: Dict, chapters: List[Dict], context: VideoContext) -> None:
    title = data.get("best_title", context.title)
    summary = str(data.get("high_impact_summary", "")).strip()
    keywords = [str(k).strip() for k in data.get("seo_keywords", []) if str(k).strip()]

    with (output_dir / "copy_paste_pack.txt").open("w", encoding="utf-8") as f:
        f.write("YOUTUBE TITLE\n")
        f.write(f"{title}\n\n")
        f.write("YOUTUBE DESCRIPTION\n")
        if summary:
            f.write(f"{summary}\n\n")
        f.write("CHAPTERS\n")
        for chapter in chapters:
            ts = chapter.get("timestamp", "").strip()
            t = chapter.get("title", "").strip()
            if ts and t:
                f.write(f"{ts} {t}\n")
        f.write("\nTAGS (COMMA SEPARATED)\n")
        f.write(", ".join(keywords) + "\n")


def write_outputs(output_dir: Path, data: Dict, context: VideoContext, chapter_style: str) -> None:
    ensure_dir(output_dir)

    chapters = normalize_chapters(data.get("chapter_suggestions", []), chapter_style=chapter_style)
    data["chapter_suggestions"] = chapters

    with (output_dir / "seo_package.json").open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    with (output_dir / "summary.md").open("w", encoding="utf-8") as f:
        f.write(f"# {data.get('best_title', context.title)}\n\n")
        f.write("## High-Impact Summary\n\n")
        f.write(str(data.get("high_impact_summary", "")).strip() + "\n")

    with (output_dir / "chapters.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "title", "what_happens", "seo_hook"])
        writer.writeheader()
        writer.writerows(chapters)

    with (output_dir / "chapters_youtube.txt").open("w", encoding="utf-8") as f:
        for c in chapters:
            ts = c.get("timestamp", "").strip()
            title = c.get("title", "").strip()
            if ts and title:
                f.write(f"{ts} {title}\n")

    keywords = [str(k).strip() for k in data.get("seo_keywords", []) if str(k).strip()]
    with (output_dir / "keywords.txt").open("w", encoding="utf-8") as f:
        for k in keywords:
            f.write(f"{k}\n")

    with (output_dir / "keywords_comma.txt").open("w", encoding="utf-8") as f:
        f.write(", ".join(keywords) + "\n")

    with (output_dir / "titles_ranked.txt").open("w", encoding="utf-8") as f:
        for i, title in enumerate(data.get("title_options", []), start=1):
            f.write(f"{i}. {title}\n")
        f.write("\nBEST TITLE:\n")
        f.write(str(data.get("best_title", "")) + "\n\n")
        f.write("WHY IT WINS:\n")
        f.write(str(data.get("best_title_reason", "")) + "\n")

    write_copy_paste_pack(output_dir, data, chapters, context)


def pick_font(size: int, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> List[str]:
    words = text.split()
    lines: List[str] = []
    current: List[str] = []

    for word in words:
        candidate = " ".join(current + [word])
        box = draw.textbbox((0, 0), candidate, font=font)
        if box[2] - box[0] <= max_width:
            current.append(word)
        else:
            if current:
                lines.append(" ".join(current))
            current = [word]

    if current:
        lines.append(" ".join(current))

    return lines


def draw_base(width: int, height: int) -> Image.Image:
    base = Image.new("RGB", (width, height), (14, 11, 8))
    draw = ImageDraw.Draw(base)
    for y in range(height):
        t = y / max(height - 1, 1)
        draw.line([(0, y), (width, y)], fill=(int(30 + 70 * t), int(18 + 60 * t), int(8 + 22 * t)))
    return base


def render_thumbnail(
    output_path: Path,
    title_text: str,
    subtitle_text: str,
    speaker_asset: Optional[Path],
    thumbnail_style: str = "ref_style_3",
) -> None:
    width, height = 1280, 720
    image = draw_base(width, height).convert("RGBA")
    draw = ImageDraw.Draw(image)

    if thumbnail_style == "ref_style_3":
        glow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        g = ImageDraw.Draw(glow)
        g.ellipse([420, -120, 1160, 540], fill=(255, 225, 50, 210))
        g.ellipse([660, 160, 1280, 760], fill=(255, 145, 24, 150))
        for x in range(0, width, 60):
            g.rectangle([x, 0, x + 8, height], fill=(255, 255, 180, 18))
        glow = glow.filter(ImageFilter.GaussianBlur(24))
        image = Image.alpha_composite(image, glow)
        draw = ImageDraw.Draw(image)

        panel = Image.new("RGBA", (760, 560), (12, 10, 8, 165))
        image.alpha_composite(panel, dest=(36, 88))
        draw.rectangle([36, 88, 52, 648], fill=(212, 102, 24, 248))

        if speaker_asset and speaker_asset.exists():
            speaker = Image.open(speaker_asset).convert("RGBA")
            speaker.thumbnail((560, 710), Image.Resampling.LANCZOS)
            alpha = speaker.split()[-1]

            subject_glow = Image.new("RGBA", speaker.size, (0, 0, 0, 0))
            sg = ImageDraw.Draw(subject_glow)
            sg.bitmap((0, 0), alpha, fill=(255, 236, 90, 240))
            subject_glow = subject_glow.filter(ImageFilter.GaussianBlur(24))

            shadow = Image.new("RGBA", speaker.size, (0, 0, 0, 0))
            sd = ImageDraw.Draw(shadow)
            sd.bitmap((0, 0), alpha, fill=(0, 0, 0, 150))
            shadow = shadow.filter(ImageFilter.GaussianBlur(20))

            x = width - speaker.size[0] - 4
            y = height - speaker.size[1] + 2
            image.alpha_composite(shadow, dest=(x + 20, y + 20))
            image.alpha_composite(subject_glow, dest=(x - 16, y - 12))
            image.alpha_composite(speaker, dest=(x, y))

        title_font = pick_font(104, bold=True)
        sub_font = pick_font(56, bold=True)

        title_lines = wrap_text(draw, title_text.upper(), title_font, max_width=680)[:3]
        y = 96
        for line in title_lines:
            fill = (255, 225, 66) if y < 210 else (255, 255, 255)
            draw.text((64, y), line, font=title_font, fill=fill, stroke_width=8, stroke_fill=(0, 0, 0))
            y += 116

        subtitle = subtitle_text.strip()
        if subtitle:
            draw.rounded_rectangle([62, 500, 600, 606], radius=28, fill=(255, 195, 62, 232), outline=(0, 0, 0, 80), width=4)
            draw.text((88, 528), subtitle.upper(), font=sub_font, fill=(26, 20, 18), stroke_width=2, stroke_fill=(255, 255, 255))
    else:
        title_font = pick_font(94, bold=True)
        lines = wrap_text(draw, title_text.upper(), title_font, max_width=760)[:3]
        y = 110
        for line in lines:
            draw.text((70, y), line, font=title_font, fill=(255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0))
            y += 102

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path, format="PNG", optimize=True)


def normalize_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def choose_speaker_asset(speaker_asset: Optional[Path], creator_name: str) -> Optional[Path]:
    if not speaker_asset:
        return None
    if speaker_asset.is_file():
        return speaker_asset
    if not speaker_asset.is_dir():
        return None

    candidates = [p for p in speaker_asset.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
    if not candidates:
        return None

    tokens = [normalize_token(tok) for tok in re.split(r"[\s_-]+", creator_name) if normalize_token(tok)]
    if tokens:
        scored = []
        for path in candidates:
            stem = normalize_token(path.stem)
            matches = sum(1 for t in tokens if t in stem)
            scored.append((matches, path.stat().st_size, path))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        if scored[0][0] > 0:
            return scored[0][2]

    return max(candidates, key=lambda p: p.stat().st_size)


def run_pipeline(
    url: Optional[str],
    video_file: Optional[Path],
    transcript_file: Optional[Path],
    episode_title: Optional[str],
    creator_name: Optional[str],
    output_dir: Path,
    openai_api_key: str,
    model: str,
    transcript_model: str,
    top_n_titles: int,
    chapter_style: str,
    speaker_asset: Optional[Path],
    thumbnail_style: str,
) -> Dict:
    client = OpenAI(api_key=openai_api_key)

    if url:
        video_id = extract_video_id(url)
        meta = fetch_video_metadata(url)
        context = VideoContext(
            video_id=video_id,
            source_ref=url,
            title=episode_title or meta["title"],
            creator=creator_name or meta["creator"],
            duration_seconds=meta.get("duration"),
        )
        segments = load_transcript_file(transcript_file) if transcript_file else load_transcript_from_youtube(video_id)
    else:
        if not video_file:
            raise ValueError("Missing --video-file")
        context = VideoContext(
            video_id=f"local-{video_file.stem}",
            source_ref=str(video_file.resolve()),
            title=episode_title or video_file.stem.replace("_", " ").replace("-", " "),
            creator=creator_name or "Unknown Creator",
            duration_seconds=None,
        )
        segments = load_transcript_file(transcript_file) if transcript_file else transcribe_video_file(client, video_file, transcript_model)

    transcript = compress_transcript(segments)
    package = generate_seo_package(client, model, context, transcript, top_n_titles)

    write_outputs(output_dir, package, context, chapter_style=chapter_style)

    thumb_data = package.get("thumbnail_copy", {}) if isinstance(package, dict) else {}
    primary = normalize_text_for_thumb(
        str(thumb_data.get("primary_text", "")),
        fallback=package.get("best_title", context.title),
        max_words=6,
    )
    secondary = normalize_text_for_thumb(
        str(thumb_data.get("secondary_text", "")),
        fallback="IN 5 MINUTES",
        max_words=4,
    )

    render_thumbnail(
        output_path=output_dir / "thumbnail.png",
        title_text=primary,
        subtitle_text=secondary,
        speaker_asset=choose_speaker_asset(speaker_asset, context.creator),
        thumbnail_style=thumbnail_style,
    )

    return {
        "output_dir": str(output_dir.resolve()),
        "best_title": package.get("best_title", context.title),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="YouTube SEO + Thumbnail generator")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="YouTube URL")
    source.add_argument("--video-file", type=Path, help="Local video file")

    parser.add_argument("--output-dir", default="output/run-001", help="Output directory")
    parser.add_argument("--transcript-file", type=Path, default=None, help="Transcript JSON file")
    parser.add_argument("--episode-title", default=None)
    parser.add_argument("--creator-name", default=None)
    parser.add_argument("--speaker-asset", type=Path, default=None, help="Speaker image file/folder")

    parser.add_argument("--openai-api-key", default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key")
    parser.add_argument("--model", default="gpt-4.1")
    parser.add_argument("--transcript-model", default="gpt-4o-mini-transcribe")
    parser.add_argument("--top-n-titles", type=int, default=10)
    parser.add_argument("--chapter-style", choices=["standard", "short", "super_short"], default="super_short")
    parser.add_argument("--thumbnail-style", choices=["ref_style_3", "viral_2026", "clean_2026"], default="ref_style_3")

    args = parser.parse_args()

    if not args.openai_api_key:
        raise SystemExit("Missing OPENAI_API_KEY")

    result = run_pipeline(
        url=args.url,
        video_file=args.video_file,
        transcript_file=args.transcript_file,
        episode_title=args.episode_title,
        creator_name=args.creator_name,
        output_dir=Path(args.output_dir),
        openai_api_key=args.openai_api_key,
        model=args.model,
        transcript_model=args.transcript_model,
        top_n_titles=args.top_n_titles,
        chapter_style=args.chapter_style,
        speaker_asset=args.speaker_asset,
        thumbnail_style=args.thumbnail_style,
    )

    print(f"Done. Outputs: {result['output_dir']}")
    print(f"Best title: {result['best_title']}")


if __name__ == "__main__":
    main()
