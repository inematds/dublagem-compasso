#!/usr/bin/env python3
"""dublagem-compasso — dubla um vídeo de personagem falando com voz clonada,
alinhando cada pedaço da fala nas batidas da boca (janelas do áudio original).

Método (validado no projeto klingaimcp, 2026-07):
  1. O áudio original do vídeo (mesmo com pronúncia ruim) é a RÉGUA DE TEMPO.
  2. Transcreve com timestamps de palavra (faster-whisper) e detecta os blocos
     de fala separados por silêncio — o "compasso" da boca.
  3. Alinha o texto correto aos blocos (difflib sobre as palavras).
  4. Gera cada bloco no chatterbox (voz clonada) em múltiplas tomadas e escolhe
     a que cabe natural na janela (ajuste fino ≤ ±20%; não coube = outra tomada).
  5. Monta a trilha nos offsets exatos, troca o áudio (vídeo intacto) e
     VERIFICA: energia presente do início ao fim de cada janela.

Uso:
  conda run -n chatterbox python3 dublar.py \
    --video clip.mp4 --texto "Frase um. Frase dois." \
    --ref ~/projetos/timesmkt3/media/voice-refs/nei4.wav --out clip-dublado.mp4

Requer: env conda `chatterbox` (chatterbox-tts + faster-whisper), ffmpeg.
"""
import argparse
import difflib
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 24000


def sh(cmd):
    subprocess.run(cmd, check=True)


def ffdur(path):
    return float(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)]).strip())


def norm(w):
    return re.sub(r"[^\wà-üÀ-Ü]", "", w.lower())


def trim_sil(x, thr_ini=0.02, thr_fim=0.05):
    """Corta silêncio das pontas. O fim usa limiar mais alto (thr_fim) porque o
    chatterbox costuma deixar um rabinho de respiração/quase-silêncio que infla
    a duração — a fala 'termina' antes do arquivo, e o fim da janela fica mudo."""
    e = np.abs(x)
    # suaviza em janelas de 20ms pra não cortar em vales entre fonemas
    k = 480
    n = len(e) // k
    env = e[:n * k].reshape(n, k).mean(axis=1)
    ini_idx = np.where(env > thr_ini * env.max())[0]
    fim_idx = np.where(env > thr_fim * env.max())[0]
    if len(ini_idx) == 0 or len(fim_idx) == 0:
        return x
    a = max(0, ini_idx[0] * k - 240)
    b = min(len(x), (fim_idx[-1] + 1) * k + 480)
    return x[a:b]


def transcrever(wav, lang):
    from faster_whisper import WhisperModel
    m = WhisperModel("medium", device="cpu", compute_type="int8")
    segs, _ = m.transcribe(str(wav), language=lang, word_timestamps=True)
    words = []
    for s in segs:
        for w in s.words:
            words.append((norm(w.word), w.start, w.end))
    return [w for w in words if w[0]]


def blocos(words, gap=0.30, min_dur=0.45):
    """Agrupa palavras em blocos de fala separados por silêncio >= gap."""
    bls = []
    for w in words:
        if bls and w[1] - bls[-1]["end"] < gap:
            bls[-1]["words"].append(w[0])
            bls[-1]["end"] = w[2]
        else:
            bls.append({"words": [w[0]], "start": w[1], "end": w[2]})
    # funde blocos curtos demais no vizinho anterior
    out = []
    for b in bls:
        if out and (b["end"] - b["start"]) < min_dur:
            out[-1]["words"] += b["words"]
            out[-1]["end"] = b["end"]
        else:
            out.append(b)
    return out


def alinhar_texto(bls, texto):
    """Distribui as palavras do texto correto pelos blocos, alinhando com as
    palavras transcritas (mesma língua/fala => alta correspondência)."""
    trans = []
    for bi, b in enumerate(bls):
        for w in b["words"]:
            trans.append((w, bi))
    alvo = [t for t in re.split(r"\s+", texto.strip()) if t]
    alvo_n = [norm(t) for t in alvo]
    sm = difflib.SequenceMatcher(a=[t[0] for t in trans], b=alvo_n, autojunk=False)
    dono = [None] * len(alvo)
    for a, b0, n in sm.get_matching_blocks():
        for k in range(n):
            dono[b0 + k] = trans[a + k][1]
    # preenche buracos com o vizinho mais próximo já atribuído
    ultimo = 0
    for i in range(len(dono)):
        if dono[i] is None:
            dono[i] = ultimo
        else:
            ultimo = dono[i]
    # garante monotonicidade
    for i in range(1, len(dono)):
        dono[i] = max(dono[i], dono[i - 1])
    por_bloco = {}
    for i, d in enumerate(dono):
        por_bloco.setdefault(d, []).append(alvo[i])
    return {bi: " ".join(ws) for bi, ws in por_bloco.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--texto", help="Texto correto da fala (ou use --texto-file)")
    ap.add_argument("--texto-file")
    ap.add_argument("--ref", required=True, help="WAV de referência da voz clonada")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang", default="pt")
    ap.add_argument("--takes", type=int, default=6)
    ap.add_argument("--gap", type=float, default=0.30, help="Silêncio (s) que separa blocos")
    ap.add_argument("--cauda", type=float, default=0.0,
                    help="Segundos de respiro no final: congela o último frame (evita corte seco)")
    ap.add_argument("--workdir", default=None)
    args = ap.parse_args()

    texto = args.texto or Path(args.texto_file).read_text()
    video = Path(args.video)
    wk = Path(args.workdir or (video.parent / f".dub-{video.stem}"))
    wk.mkdir(parents=True, exist_ok=True)
    vdur = ffdur(video)

    print(f"[1/5] extraindo áudio-guia de {video.name} ({vdur:.2f}s)")
    guia = wk / "guia.wav"
    sh(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-vn",
        "-ar", str(SR), "-ac", "1", str(guia)])

    print("[2/5] transcrevendo (timestamps de palavra)…")
    words = transcrever(guia, args.lang)
    if not words:
        sys.exit("nenhuma fala detectada no vídeo")
    bls = blocos(words, gap=args.gap)
    textos = alinhar_texto(bls, texto)
    for bi, b in enumerate(bls):
        print(f"  bloco {bi}: {b['start']:.2f}-{b['end']:.2f}s "
              f"boca='{' '.join(b['words'])}' -> fala='{textos.get(bi, '')}'")

    print("[3/5] gerando tomadas (chatterbox)…")
    import torch
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)

    pecas = []
    for bi, b in enumerate(bls):
        txt = textos.get(bi, "").strip()
        if not txt:
            continue
        wdur = b["end"] - b["start"]
        cands = []
        for t in range(args.takes):
            wav = model.generate(txt, language_id=args.lang,
                                 audio_prompt_path=args.ref,
                                 exaggeration=0.5, cfg_weight=0.65,
                                 temperature=0.75)
            x = trim_sil(wav.squeeze().cpu().numpy())
            dur = len(x) / SR
            tempo = dur / wdur
            tail = np.abs(x[-int(0.5 * SR):]).mean()
            # tomada com cauda muda é REJEITADA (pen alto), não só penalizada:
            # é ela que produz "boca aberta sem som" no fim da janela.
            pen = abs(tempo - 1.0) + (0 if 0.80 <= tempo <= 1.30 else 10) \
                + (10 if tail < 0.008 else 0)
            cands.append((pen, x, dur))
            print(f"  bloco {bi} take{t + 1}: {dur:.2f}s (janela {wdur:.2f}s, tempo {tempo:.2f})")
            if pen < 0.12:
                break  # tomada excelente, não gasta GPU à toa
        cands.sort(key=lambda c: c[0])
        _, x, dur = cands[0]
        p = wk / f"bloco{bi}.wav"
        sf.write(str(p), x, SR)
        pecas.append((str(p), b["start"], wdur, dur))

    print("[4/5] montando trilha e trocando o áudio…")
    inputs, filt, mix = [], [], []
    for i, (p, ws, wd, dur) in enumerate(pecas):
        tempo = max(0.80, min(1.30, dur / wd))
        delay = int(ws * 1000)
        inputs += ["-i", p]
        filt.append(f"[{i}:a]atempo={tempo:.4f},adelay={delay}|{delay}[s{i}]")
        mix.append(f"[s{i}]")
    fc = (";".join(filt) + ";" + "".join(mix) +
          f"amix=inputs={len(pecas)}:normalize=0,apad=whole_dur={vdur},"
          f"atrim=0:{vdur},loudnorm=I=-16:TP=-1.5[out]")
    trilha = wk / "trilha.wav"
    sh(["ffmpeg", "-y", "-v", "error"] + inputs +
       ["-filter_complex", fc, "-map", "[out]", "-ar", "44100", str(trilha)])
    if args.cauda > 0:
        # congela o último frame por --cauda s (respiro; evita final duro).
        # exige re-encode do vídeo (tpad), áudio ganha silêncio no rabo (apad).
        sh(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-i", str(trilha),
            "-filter_complex",
            f"[0:v]tpad=stop_mode=clone:stop_duration={args.cauda}[v];"
            f"[1:a]apad=pad_dur={args.cauda}[a]",
            "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k", args.out])
    else:
        sh(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-i", str(trilha),
            "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
            "-b:a", "192k", args.out])

    print("[5/5] verificando (boca aberta sem som é o defeito nº 1)…")
    out = subprocess.check_output(
        ["ffmpeg", "-v", "error", "-i", str(trilha),
         "-f", "f32le", "-ac", "1", "-ar", "1000", "-"])
    xx = np.abs(np.frombuffer(out, dtype=np.float32))
    falhas = 0
    for i, (p, ws, wd, dur) in enumerate(pecas):
        we = ws + wd
        st = xx[int(ws * 1000):int((ws + 0.4) * 1000)].mean()
        tl = xx[int(max(ws, we - 0.4) * 1000):int(we * 1000)].mean()
        ok = st > 0.005 and tl > 0.004
        print(f"  janela {i} [{ws:.2f}-{we:.2f}]: início={st:.4f} fim={tl:.4f} "
              f"{'OK' if ok else 'FALHOU'}")
        if not ok:
            falhas += 1
    if falhas:
        sys.exit(f"VERIFICAÇÃO FALHOU em {falhas} janela(s) — não entregue este resultado; "
                 "rode de novo com mais --takes ou ajuste o texto (reticências alongam).")
    print(f"PRONTO: {args.out}")


if __name__ == "__main__":
    main()
