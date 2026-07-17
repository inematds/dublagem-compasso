# dublagem-compasso

Dubla um vídeo de **personagem falando** (gerado no Kling, Veo, Seedance…) com uma **voz clonada local** (chatterbox), alinhando cada pedaço da fala nas **batidas da boca** — o "compasso" — em vez de só colar uma narração por cima.

## O problema que resolve

Geradores de vídeo com IA falam idiomas não-ingleses com pronúncia ruim (PT-BR sai com sotaque e palavras trocadas), e **não aceitam áudio de entrada** — não dá pra mandar sua voz. Trocar o áudio por uma narração TTS comum quebra o sincronismo: boca abrindo sem som, fala atravessando pausas.

## O método (validado em produção, 2026-07)

1. **O áudio original é a régua de tempo.** Gere o vídeo com o prompt inteiro no idioma alvo + a fala entre aspas + áudio nativo ligado — mesmo ruim, ele faz a boca articular as frases certas nos tempos certos.
2. **Transcrição com timestamps de palavra** (faster-whisper) → blocos de fala separados por silêncio = o compasso da boca.
3. **Alinhamento do texto correto aos blocos** (difflib — mesma língua e mesma fala garantem alta correspondência).
4. **TTS multi-tomada por bloco** (chatterbox + voz clonada): gera N tomadas e escolhe a que cabe **natural** na janela (ajuste fino ≤ ±20%; não coube = outra tomada, nunca esticão).
5. **Montagem nos offsets exatos** + troca do áudio (vídeo intacto, `-c:v copy`).
6. **Verificação automática**: energia de fala presente do início ao fim de cada janela — *boca aberta sem som* é o defeito nº 1 e o script falha alto se detectar.

## Uso

```bash
conda run -n chatterbox python3 dublar.py \
  --video clip.mp4 \
  --texto "Não adianta ficar só assistindo. Você precisa praticar." \
  --ref ~/vozes/minha-voz.wav \
  --out clip-dublado.mp4 \
  --cauda 2          # opcional: congela o último frame por 2s (evita final duro)
```

Opções úteis:
- `--takes 6` — tomadas por bloco (mais tomadas = melhor encaixe, mais GPU)
- `--gap 0.30` — silêncio (s) que separa blocos de fala
- `--lang pt` — idioma da transcrição e do TTS
- `--texto-file roteiro.txt` — texto vindo de arquivo

## Dicas de calibragem

- Se um bloco da boca é **lento** demais pro texto (fala pausada no original), escreva o texto com **reticências** ("ler... testar... errar...") — o TTS reproduz o ritmo pausado.
- A referência de voz: WAV 24kHz mono, ~10s de fala limpa (o chatterbox usa só os primeiros 10s).
- Close extremo de boca exige lip-sync por fonema (ex.: HeyGen); este método alinha por batida de frase/pedaço — excelente pra plano médio, aceitável em close.

## Requisitos

- env conda `chatterbox` com `chatterbox-tts` e `faster-whisper`
- `ffmpeg` no PATH
- GPU ajuda (TTS); a transcrição roda em CPU

## Origem

Método desenvolvido e validado no projeto [klingaimcp](https://github.com/inematds/klingaimcp) (avatar falante do Kling dublado em PT-BR com voz clonada).
