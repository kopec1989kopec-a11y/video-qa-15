#!/usr/bin/env python3
"""QA analysis of 15 YouTube videos: download + objective video/audio metrics.
Results land in results/ (JSON + frames) and are pushed back to the repo.
"""
import json, os, re, subprocess, sys, shutil, glob, math, traceback

import numpy as np

VIDEOS = [
    "AUMxF53wXf0", "2m066BN5ZW8", "drsAUlJULqo", "tRJSp75381M", "2dWkuuxdE3E",
    "VoWPYdfylHA", "3-LQ2UJ42OU", "5D7MmVtl4k4", "NYROy3ii-gg", "TS5wfvfFXCU",
    "1La4QzGeaaQ", "3VJEM2gHMmE", "ksYMQcqDQZU", "D8zHRcYt_kM", "hrROEBdxfow",
]
WORK = "/tmp/work"
RES = "results"
CLIENT_SETS = [
    "",                                   # default
    "youtube:player_client=android",
    "youtube:player_client=web_embedded",
    "youtube:player_client=ios",
    "youtube:player_client=tv",
]

def sh(cmd, timeout=1800, capture=True):
    return subprocess.run(cmd, shell=True, timeout=timeout,
                          stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.STDOUT if capture else None, text=True)

def ytdlp_info(vid, log):
    url = f"https://www.youtube.com/watch?v={vid}"
    for cs in CLIENT_SETS:
        args = f'--extractor-args "{cs}"' if cs else ""
        r = sh(f'yt-dlp {args} --no-playlist --skip-download --print-json "{url}" > /tmp/info.json 2>/tmp/info.err', timeout=300)
        try:
            with open("/tmp/info.json") as f:
                return json.load(f), cs
        except Exception:
            err = ""
            try: err = open("/tmp/info.err").read()[-1500:]
            except Exception: pass
            log.append(f"info fail client={cs or 'default'}: {err.strip()[-400:]}")
    return None, None

def ytdlp_download(vid, duration, cs, log):
    """Download full video if short, else evenly spaced sections."""
    out = f"{WORK}/{vid}/src"
    os.makedirs(out, exist_ok=True)
    fmt = 'bv*[height<=1080][ext=mp4]+ba[ext=m4a]/bv*[height<=1080]+ba/b[height<=1080]/b'
    base = f'yt-dlp --no-playlist -f "{fmt}" --merge-output-format mp4 --retries 3 --fragment-retries 3'
    if cs: base += f' --extractor-args "{cs}"'
    url = f"https://www.youtube.com/watch?v={vid}"
    sections = ""
    if duration and duration > 3600:
        pts = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]
        secs = []
        for p in pts:
            s = int(duration * p); e = min(int(duration * p + 120), int(duration))
            secs.append(f'"*{s}-{e}"')
        sections = " --download-sections " + " ".join(secs)
    elif duration and duration > 1500:
        pts = [0.0, 0.25, 0.5, 0.75, 0.95]
        secs = []
        for p in pts:
            s = int(duration * p); e = min(int(duration * p + 180), int(duration))
            secs.append(f'"*{s}-{e}"')
        sections = " --download-sections " + " ".join(secs)
    clients = [cs] if cs else []
    clients += [c for c in CLIENT_SETS if c and c != cs]
    for c in clients:
        b2 = base
        if c: b2 = f'yt-dlp --extractor-args "{c}" --no-playlist -f "{fmt}" --merge-output-format mp4 --retries 3 --fragment-retries 3'
        r = sh(f'{b2}{sections} -o "{out}/%(id)s.%(ext)s" "{url}" 2>&1 | tail -5', timeout=3000)
        files = glob.glob(f"{out}/*.mp4") + glob.glob(f"{out}/*.webm") + glob.glob(f"{out}/*.mkv")
        if files and os.path.getsize(files[0]) > 100000:
            return files[0], c
        log.append(f"dl fail client={c or 'default'}: {r.stdout.strip()[-300:] if r.stdout else ''}")
    return None, None

def ffprobe(path):
    r = sh(f'ffprobe -v quiet -print_format json -show_format -show_streams "{path}"')
    try: return json.loads(r.stdout)
    except Exception: return {}

def extract_frames(vid, video, duration, n_want=36):
    import cv2
    fdir = f"{RES}/frames/{vid}"
    os.makedirs(fdir, exist_ok=True)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = total / fps if fps else 0
    step = max(1, int(total / (n_want + 1))) if total else int(fps * 10)
    frames, ts_list = [], []
    idx = step // 2
    while idx < total and len(frames) < n_want:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        if ok and fr is not None:
            ts = idx / fps
            frames.append((ts, fr))
            ts_list.append(round(ts, 1))
        idx += step
    cap.release()
    saved = []
    for i, (ts, fr) in enumerate(frames):
        h, w = fr.shape[:2]
        scale = min(1.0, 1280 / w)
        out = cv2.resize(fr, (int(w * scale), int(h * scale))) if scale < 1 else fr
        p = f"{fdir}/f{i:03d}_{int(ts)}s.jpg"
        cv2.imwrite(p, out, [cv2.IMWRITE_JPEG_QUALITY, 88])
        saved.append(p)
    crops = []
    if frames:
        for j in (len(frames) // 3, 2 * len(frames) // 3):
            ts, fr = frames[j]
            h, w = fr.shape[:2]
            ch, cw = min(500, h), min(700, w)
            y0, x0 = (h - ch) // 2, (w - cw) // 2
            crop = fr[y0:y0 + ch, x0:x0 + cw]
            p = f"{fdir}/crop{j:03d}_{int(ts)}s_1to1.jpg"
            cv2.imwrite(p, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            crops.append(p)
    return frames, ts_list, dur, saved, crops

def blockiness(gray):
    import cv2
    g = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) ** 2 + cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) ** 2
    h, w = g.shape
    col = g.mean(axis=0); row = g.mean(axis=1)
    on_v = [col[x] for x in range(8, w - 8, 8)]
    off_v = [col[x] for x in range(4, w - 8, 8) if x % 8 != 0]
    on_h = [row[y] for y in range(8, h - 8, 8)]
    off_h = [row[y] for y in range(4, h - 8, 8) if y % 8 != 0]
    def ratio(on, off):
        on, off = np.mean(on), np.mean(off)
        return float(on / off) if off > 1e-6 else 1.0
    return (ratio(on_v, off_v) + ratio(on_h, off_h)) / 2

def posterization(gray):
    import cv2
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    mask = np.abs(lap) < 3
    if mask.sum() < gray.size * 0.05:
        return None
    vals = gray[mask]
    hist = np.bincount(vals, minlength=256)
    nonzero = int((hist > 0).sum())
    peaks = int((hist > hist.mean() * 3).sum())
    return {"levels_in_flat": nonzero, "strong_peaks": peaks}

def analyze_video_metrics(vid, frames, ts_list):
    import cv2
    m = {"frame_count": len(frames)}
    if not frames: return m
    hists = []
    for ts, fr in frames:
        hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
        hst = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hst, hst)
        hists.append(hst)
    cuts = []
    for i in range(1, len(hists)):
        d = cv2.compareHist(hists[i - 1], hists[i], cv2.HISTCMP_BHATTACHARYYA)
        if d > 0.55:
            cuts.append(round(ts_list[i], 1))
    m["cuts_detected"] = cuts
    if len(cuts) > 1:
        gaps = [round(b - a, 1) for a, b in zip(cuts, cuts[1:])]
        m["min_cut_gap_s"] = min(gaps)
        m["cut_gaps_lt5s"] = len([g for g in gaps if g < 5])
    bounds = [0.0] + cuts + [ts_list[-1]]
    m["longest_no_cut_s"] = round(max(b - a for a, b in zip(bounds, bounds[1:])), 1)
    lap_vars, overexp, blocks, postr = [], [], [], []
    for ts, fr in frames[::2]:
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        lap_vars.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        overexp.append(float((gray > 245).mean() * 100))
        blocks.append(blockiness(gray))
        p = posterization(gray)
        if p: postr.append(p["levels_in_flat"])
    m["laplacian_var_median"] = round(float(np.median(lap_vars)), 1)
    m["overexposed_pct_median"] = round(float(np.median(overexp)), 2)
    m["overexposed_pct_max"] = round(float(max(overexp)), 2)
    m["blockiness_median"] = round(float(np.median(blocks)), 3)
    m["blockiness_max"] = round(float(max(blocks)), 3)
    if postr:
        m["flat_levels_median"] = int(np.median(postr))
    return m

def analyze_audio(vid, video):
    adir = f"{WORK}/{vid}"
    wav = f"{adir}/audio.wav"
    sh(f'ffmpeg -y -v error -i "{video}" -vn -ac 1 -ar 16000 "{wav}"')
    if not os.path.exists(wav) or os.path.getsize(wav) < 1000:
        return {"has_audio": False}
    r = {"has_audio": True}
    ln = sh(f'ffmpeg -hide_banner -i "{wav}" -af loudnorm=print_format=json -f null - 2>&1 | tail -20')
    m = re.search(r'\{[^{}]*"input_i"[^{}]*\}', ln.stdout or "", re.S)
    if m:
        try:
            j = json.loads(m.group(0))
            r["loudness_i_lufs"] = float(j["input_i"])
            r["true_peak_dbtp"] = float(j["input_tp"])
            r["loudness_range_lu"] = float(j["input_lra"])
        except Exception: pass
    sd = sh(f'ffmpeg -hide_banner -i "{wav}" -af silencedetect=noise=-40dB:d=2 -f null - 2>&1')
    sil = re.findall(r'silence_(?:start|end): ([0-9.]+)', sd.stdout or "")
    total_sil = 0.0
    starts = [float(sil[i]) for i in range(0, len(sil), 2)]
    ends = [float(sil[i]) for i in range(1, len(sil), 2)]
    dur = 0.0
    pr = sh(f'ffprobe -v quiet -show_entries format=duration -of csv=p=0 "{wav}"')
    try: dur = float((pr.stdout or "0").strip())
    except Exception: pass
    for s, e in zip(starts, ends):
        total_sil += e - s
    if dur:
        r["audio_duration_s"] = round(dur, 1)
        r["silence_pct"] = round(total_sil / dur * 100, 1)
    try:
        import wave
        with wave.open(wav) as wf:
            n = wf.getnframes()
            data = np.frombuffer(wf.readframes(n), dtype=np.int16).astype(np.float32) / 32768
        r["rms_dbfs"] = round(float(20 * np.log10(np.sqrt((data ** 2).mean()) + 1e-10)), 1)
        peak = float(np.abs(data).max())
        r["peak_dbfs"] = round(20 * math.log10(peak + 1e-10), 1)
        r["clipped_samples_pct"] = round(float((np.abs(data) > 0.999).mean() * 100), 4)
        sr = 16000; win = sr
        flux = []
        prev = None
        for i in range(0, len(data) - win, win):
            seg = data[i:i + win] * np.hanning(win)
            sp = np.abs(np.fft.rfft(seg))
            if prev is not None:
                flux.append(float(np.mean(np.abs(sp - prev))))
            prev = sp
        if flux:
            r["spectral_flux_mean"] = round(float(np.mean(flux)), 4)
            r["spectral_flux_var"] = round(float(np.var(flux)), 4)
        zcr = float(np.mean(np.abs(np.diff(np.signbit(data[::10])))))
        r["zcr"] = round(zcr, 4)
    except Exception as e:
        r["spectral_error"] = str(e)[:100]
    return r

def main():
    os.makedirs(RES, exist_ok=True)
    summary = {}
    for k, vid in enumerate(VIDEOS, 1):
        log = []
        rec = {"n": k, "video_id": vid, "url": f"https://www.youtube.com/watch?v={vid}", "log": log}
        vdir = f"{WORK}/{vid}"
        os.makedirs(vdir, exist_ok=True)
        try:
            info, cs = ytdlp_info(vid, log)
            if info:
                rec["title"] = info.get("title")
                rec["duration_s"] = info.get("duration")
                rec["channel"] = info.get("channel")
                rec["upload_date"] = info.get("upload_date")
                rec["was_live"] = bool(info.get("was_live"))
                rec["live_status"] = info.get("live_status")
                fmts = sorted({f.get("height") for f in info.get("formats", []) if f.get("height")}, reverse=True)
                rec["available_heights"] = fmts[:8]
                rec["fps_list"] = sorted({f.get("fps") for f in info.get("formats", []) if f.get("fps") and f.get("height") and f["height"] >= 1080}, reverse=True)[:4]
                rec["has_audio_track"] = any(f.get("acodec") not in (None, "none") for f in info.get("formats", []))
                rec["categories"] = info.get("categories")
                rec["age_limit"] = info.get("age_limit")
            else:
                rec["error"] = "info_unavailable"
                summary[vid] = rec
                continue
            video, used_client = ytdlp_download(vid, info.get("duration"), cs, log)
            rec["download_client"] = used_client
            if not video:
                rec["error"] = "download_failed"
                summary[vid] = rec
                continue
            pr = ffprobe(video)
            vs = next((s for s in pr.get("streams", []) if s.get("codec_type") == "video"), {})
            rec["actual_width"] = vs.get("width"); rec["actual_height"] = vs.get("height")
            rec["actual_fps"] = vs.get("avg_frame_rate")
            rec["vcodec"] = vs.get("codec_name")
            rec["bitrate_kbps"] = round(int(pr.get("format", {}).get("bit_rate", 0)) / 1000)
            rec["file_mb"] = round(os.path.getsize(video) / 1e6, 1)
            rec["downloaded_duration_s"] = round(float(pr.get("format", {}).get("duration", 0)), 1)
            frames, ts_list, dur, saved, crops = extract_frames(vid, video, info.get("duration"))
            rec["frame_timestamps"] = ts_list
            rec["video_metrics"] = analyze_video_metrics(vid, frames, ts_list)
            rec["audio_metrics"] = analyze_audio(vid, video)
            import cv2
            cb = []
            for c in crops:
                g = cv2.imread(c, cv2.IMREAD_GRAYSCALE)
                if g is not None: cb.append(round(blockiness(g), 3))
            rec["blockiness_native_crops"] = cb
            shutil.rmtree(vdir, ignore_errors=True)
        except Exception:
            rec["error"] = "exception"
            rec["exception"] = traceback.format_exc()[-800:]
        summary[vid] = rec
        with open(f"{RES}/{vid}.json", "w") as f:
            json.dump(rec, f, indent=1, ensure_ascii=False)
    with open(f"{RES}/summary.json", "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    print("DONE")

if __name__ == "__main__":
    main()
