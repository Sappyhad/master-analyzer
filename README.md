<div align="center">

# 🎚️ MasterCheck

**Lokalny analizator miksu i masteringu audio — fakty pomiarowe oddzielone od hipotez.**

[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?logo=scipy&logoColor=white)](https://scipy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?logo=plotly&logoColor=white)](https://matplotlib.org/)
[![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![FFmpeg](https://img.shields.io/badge/FFmpeg-optional-007808?logo=ffmpeg&logoColor=white)](https://ffmpeg.org/)
[![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)](https://docs.pytest.org/)

[![BS.1770-4](https://img.shields.io/badge/ITU--R-BS.1770--4-orange)](https://www.itu.int/rec/R-REC-BS.1770)
[![EBU R128](https://img.shields.io/badge/EBU-R128-orange)](https://tech.ebu.ch/publications/r128)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)](#2-instalacja)
[![Last commit](https://img.shields.io/github/last-commit/Sappyhad/master-analyzer)](https://github.com/Sappyhad/master-analyzer/commits)
[![Top language](https://img.shields.io/github/languages/top/Sappyhad/master-analyzer)](https://github.com/Sappyhad/master-analyzer)

</div>

Mierzy głośność (BS.1770-4 / EBU R128), true peak, dynamikę, balans tonalny, stereo/fazę, clipping,
symuluje normalizację streamingową i kodeki stratne, porównuje utwór z referencjami i generuje diagnozę,
która **oddziela fakty pomiarowe od hipotez**.

> [!NOTE]
> MasterCheck wspiera decyzje — nie zastępuje odsłuchu na skalibrowanym torze. Nie ocenia mastera
> jako „dobry/zły” i nie udaje, że zna gain reduction limitera.

## ⚡ Szybki start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py --input "input/my_master.wav"
```

Wynik ląduje w `output/<nazwa>_<timestamp>/report.html`.

## ✨ Funkcje

| | |
|---|---|
| 🔊 **Głośność** | LUFS-I / S / M, LRA, true peak (oversampling), RMS, crest, PLR |
| 📈 **Dynamika** | crest w oknach, gęstość szczytów, sekcje energii, najgłośniejszy fragment, pumping |
| 🎛️ **Spektrum** | Welch PSD, energia pasm, tilt, rezonanse, wykrycie low-passa kodeka |
| 🎧 **Stereo / faza** | korelacja, M/S, balans L/R, mono fold-down, regiony fazowe |
| ⚠️ **Clipping** | rozróżnienie *detected* vs *suspected*, prominencja transjentów |
| 📡 **Streaming** | symulacja normalizacji i realne kodowanie MP3/AAC przez FFmpeg |
| 🆚 **Referencje** | porównanie surowe, po wyrównaniu głośności i najgłośniejszych fragmentów |
| 📄 **Raport** | `report.html`, `analysis.json`, `metrics.csv`, wykresy PNG |

## 📚 Spis treści

1. [Architektura](#1-architektura)
2. [Instalacja](#2-instalacja)
3. [Uruchomienie](#3-uruchomienie)
4. [Co i jak jest mierzone](#4-co-i-jak-jest-mierzone)
5. [Porównanie z referencjami](#5-porównanie-z-referencjami)
6. [Jak czytać raport](#6-jak-czytać-raport)
7. [Konfiguracja](#7-konfiguracja-configjson)
8. [Obsługa błędów](#8-obsługa-błędów)

---

## 1. Architektura

```
master-analyzer/
├── main.py                  # punkt wejścia, sprawdza zależności
├── config.json              # progi, pasma, kodeki, profile gatunkowe
├── requirements.txt
├── input/  reference/  output/
├── mastercheck/
│   ├── audio_loader.py      # dekodowanie (soundfile / FFmpeg), walidacja, limity pamięci
│   ├── loudness.py          # K-weighting, LUFS-I/S/M, LRA, true peak, RMS, crest, PLR
│   ├── dynamics.py          # crest w oknach, gęstość szczytów, sekcje, najgłośniejszy fragment, pumping
│   ├── spectrum.py          # Welch PSD, energia pasm, tilt, rezonanse, wykrycie low-passa kodeka
│   ├── stereo.py            # korelacja, M/S, balans L/R, mono fold-down, regiony fazowe
│   ├── transients.py        # clipping wykryty vs podejrzewany, prominencja transjentów
│   ├── normalization.py     # symulacja normalizacji (gain-only / peak-limited)
│   ├── codec_simulation.py  # realne kodowanie MP3/AAC przez FFmpeg i ponowny pomiar
│   ├── analyzer.py          # orkiestracja jednego pliku, metryki nagłówkowe
│   ├── comparison.py        # porównanie z referencjami: surowe, po wyrównaniu głośności, fragmenty
│   ├── diagnosis.py         # reguły diagnostyczne + odpowiedzi na kluczowe pytania
│   ├── plots.py             # wykresy PNG
│   ├── report.py            # report.html, analysis.json, metrics.csv, podsumowanie w terminalu
│   ├── models.py, config.py, errors.py, cli.py
└── tests/                   # pytest: zgodność z BS.1770/EBU, loader, metryki, diagnoza
```

Zmiana względem proponowanej struktury: `src/` → pakiet `mastercheck/` (czyste importy
`from mastercheck.loudness import ...`, możliwość uruchamiania testów i CLI bez kombinowania z `sys.path`),
plus osobne moduły `normalization.py`, `analyzer.py`, `plots.py`, `cli.py`.

### Dlaczego te biblioteki

| Biblioteka | Po co | Uwagi |
|---|---|---|
| numpy / scipy | FFT, filtry IIR (K-weighting), polifazowy oversampling (true peak), Welch, STFT | standard DSP w Pythonie |
| soundfile (libsndfile) | WAV 16/24/32/float, FLAC, AIFF, OGG, MP3 | dekoduje do float64 bez obcinania overów |
| pyloudnorm | niezależna kontrola LUFS-I | własna implementacja BS.1770-4 jest sprawdzana względem niej |
| matplotlib | wykresy PNG | backend Agg, bez GUI |
| pandas | eksport CSV | |
| FFmpeg (systemowy lub `imageio-ffmpeg`) | M4A/AAC, symulacja kodeków | opcjonalny; bez niego reszta działa |

**librosa celowo pominięta** — ciężka (numba/llvmlite, częste problemy instalacji na Windows),
a wszystko co potrzebne (STFT, filtry, resampling) daje scipy.

---

## 2. Instalacja

Wymagany Python **3.10+**.

### Windows (PowerShell)

```powershell
cd master-analyzer
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

FFmpeg: `imageio-ffmpeg` z requirements dostarcza gotową binarkę — nic więcej nie trzeba.
Alternatywnie systemowy: `winget install Gyan.FFmpeg` (musi być w PATH).

### macOS / Linux

```bash
cd master-analyzer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# opcjonalnie systemowy FFmpeg:
brew install ffmpeg            # macOS
sudo apt install ffmpeg        # Debian/Ubuntu
```

Na Linuksie, jeśli `soundfile` zgłasza brak libsndfile: `sudo apt install libsndfile1`.

### Testy

```bash
python -m pytest -q
```

---

## 3. Uruchomienie

```bash
# podstawowo — automatycznie użyje plików z reference/, jeśli są
python main.py --input "input/my_master.wav"

# inny folder referencji
python main.py --input "input/my_master.wav" --reference-dir "D:/refs/techno"

# tylko fragment 00:45–01:15 (sekundy lub mm:ss)
python main.py --input "input/my_master.wav" --start 45 --end 75
python main.py --input "input/my_master.wav" --start 00:45 --end 01:15

# porównanie konkretnych zakresów: utwór 45–75 s vs referencje 60–90 s
python main.py --input "input/my_master.wav" --start 45 --end 75 --ref-start 60 --ref-end 90

# cel symulacji normalizacji i sufit true peak
python main.py --input "input/my_master.wav" --target-lufs -14 --tp-ceiling -1

# szybko: bez referencji, bez kodeków, bez wykresów
python main.py --input "input/my_master.wav" --no-reference --no-codec --no-plots

# profil gatunkowy (tylko dla heurystyk bez referencji), własny config
python main.py --input "input/my_master.wav" --genre edm --config my_thresholds.json
```

Wszystkie opcje: `python main.py --help`.

| Opcja | Znaczenie |
|---|---|
| `--input` | plik do analizy (wymagany) |
| `--reference-dir` | folder referencji (domyślnie `reference/`) |
| `--no-reference` | pomiń referencje |
| `--start`, `--end` | zakres analizy utworu |
| `--ref-start`, `--ref-end` | zakres analizy każdej referencji |
| `--target-lufs` | cel symulacji normalizacji |
| `--tp-ceiling` | sufit true peak w symulacji normalizacji |
| `--tp-limit` | próg ostrzeżenia true peak w diagnozie |
| `--genre` | profil z `config.json → genre_profiles` |
| `--no-codec`, `--no-plots` | pomiń kodeki / wykresy |
| `--output-dir` | folder nadrzędny wyników |
| `--config` | dodatkowy JSON nadpisujący `config.json` |

### Wynik

Każde uruchomienie tworzy `output/<nazwa>_<RRRRMMDD_GGMMSS>/`:

- `report.html` — pełny raport (podsumowanie, diagnozy, tabele, wykresy, ograniczenia)
- `analysis.json` — wszystkie dane, łącznie z przebiegami LUFS-M/S i korelacji w czasie
- `metrics.csv` — tabela metryk: utwór, każda referencja, mediana, różnica
- `waveform.png`, `loudness.png`, `spectrum.png`, `spectrogram.png`, `dynamics.png`, `stereo.png`
- `comparison_*.png` — jeśli są referencje

Oryginalne pliki audio są tylko czytane. Symulacja kodeków pracuje na kopii tymczasowej.

---

## 4. Co i jak jest mierzone

Każda metryka w raporcie ma jednostkę, metodę i status: **standard** (implementacja normy),
**approximation** (metoda o znanych ograniczeniach), **derived** (prosta arytmetyka, definicja podana),
**heuristic** (wskaźnik, nie wielkość fizyczna), **not_available** (nie dało się wiarygodnie policzyć — bez zmyślania wartości).

| Metryka | Metoda | Status |
|---|---|---|
| LUFS-I | BS.1770-4: K-weighting, bloki 400 ms / 75 % overlap, bramki −70 LUFS i −10 LU. Wartość z pyloudnorm, krzyżowo z własną implementacją | standard |
| LUFS-M / LUFS-S | okna 400 ms / 3 s, odświeżanie 10 Hz (EBU Tech 3341) | standard |
| LRA | EBU Tech 3342: bloki 3 s, bramki −70 LUFS i −20 LU, percentyl 95 − 10 | standard |
| True peak | oversampling 4× (dla 44.1/48 kHz; 2× dla 88.2/96 kHz), FIR Kaiser 48 tapów/fazę | approximation — może zaniżać rzeczywiste szczyty międzypróbkowe (BS.1770-4: do ~0.7 dB dla 4× w najgorszym przypadku) |
| Sample peak | max |próbka| | standard — **to nie jest true peak** |
| RMS | nieważony, wszystkie kanały, cały zakres; sinus 0 dBFS = −3.01 dBFS (bez offsetu AES17) | derived |
| Crest factor | sample peak − RMS (dB), cały plik + mediana w oknach 400 ms | derived |
| PLR | true peak − LUFS-I | derived |
| Korelacja | Σ L·R / √(ΣL²·ΣR²), cały plik, w oknach 1 s i dla pasma < 150 Hz | derived |
| Side/Mid | 20·log10(RMS(S)/RMS(M)), M=(L+R)/2, S=(L−R)/2 | derived |
| Mono fold-down | LUFS-I [M,M] − LUFS-I [L,R] | derived |
| Pasma | całka z PSD Welcha (Hann, 50 %) sygnału mid; % energii 20 Hz–20 kHz i dB | derived |

Ograniczenia, które raport podaje wprost:

- **Gain reduction limitera jest niemierzalny** z gotowego pliku stereo. Wskaźniki (crest, gęstość szczytów, PLR, LRA, prominencja transjentów) opisują efekt, nie proces.
- **Clipping**: „detected” = próbki > 0 dBFS (pliki float) lub powtarzające się płaskie odcinki na pełnej skali z ostrymi krawędziami. „suspected” = płaskie wierzchołki poniżej 0 dBFS (np. clipping + późniejsze ściszenie albo celowy clipper). Na którym etapie produkcji powstał — nie da się udowodnić.
- **Sekcje** to poziomy energii (klasteryzacja LUFS-S), nie rozpoznane zwrotki/refreny.
- **Pumping** to słaba wskazówka statystyczna (ujemna korelacja fluktuacji obwiedni < 120 Hz vs 2–8 kHz).
- **Normalizacja** i **kodeki** to symulacje, nie odwzorowanie Spotify/Apple Music/YouTube.

---

## 5. Porównanie z referencjami

Dla każdej referencji raport pokazuje trzy widoki:

1. **Surowe wartości** — utwór, referencja, różnica (tabela z sekcji 8 specyfikacji + dodatkowe metryki).
2. **Po wyrównaniu głośności** — referencja jest *wirtualnie* przesunięta o (LUFS-I utworu − LUFS-I referencji) dB,
   tym samym miernikiem BS.1770-4 dla obu plików. Zastosowany gain jest pokazany. Poziomy pasm, TP, RMS
   porównywane są przy równej głośności — różnice tonalne nie mylą się z głośnością odtwarzania.
   Pliki nie są modyfikowane: przesunięcie gainem przesuwa każdą wartość w dB o tyle samo, więc to dokładna arytmetyka.
3. **Najgłośniejszy fragment** — okno 15 s (konfigurowalne) o najwyższej energii LUFS-M w każdym pliku.

Przy kilku referencjach: wyniki osobno dla każdej + mediana/min/max i flaga „poza zakresem referencji”.

---

## 6. Jak czytać raport

1. **Summary** — odpowiedzi na kluczowe pytania (bezpieczny peak? dynamika vs referencje? balans tonalny?
   clipping? zachowanie po normalizacji? co sprawdzić najpierw? czego nie da się stwierdzić?).
2. **Diagnosis** — każda diagnoza ma:
   - typ: **measured problem** (mierzalny fakt), **potential issue** (hipoteza), **observation** (informacja),
   - **confidence**: low / medium / high,
   - **Evidence** — zmierzone wartości,
   - **Possible causes** — hipotezy,
   - **Suggested tests** — co odsłuchać/sprawdzić,
   - **Recommended action** i **May make it worse** — czego unikać,
   - **Missing information** — czego brakuje do pewnej diagnozy.
3. Tabele i wykresy dla każdej dziedziny.

Praktyczne wskazówki:

- Wysoki true peak ≠ zły master. To ryzyko przy kodowaniu stratnym — sprawdź sekcję kodeków i tryb ISP limitera.
- Niski LUFS-I vs referencje: najpierw porównaj przy wyrównanej głośności (tabela „matched”). Często różnica to środek pasma, transjenty lub nadmiar basu zjadający headroom — nie „za mało limitera”.
- Wysoki LUFS-I, a master brzmi słabo: patrz balans tonalny, clipping i prominencję transjentów — nie dokładaj limitowania.
- Korelacja bliska +1 to nie wada; szerokie stereo to nie zaleta. Liczy się mono fold-down i niskie pasmo.
- Progi w `config.json` to punkty startowe — dostosuj do gatunku.

---

## 7. Konfiguracja (`config.json`)

- `analysis` — oversampling true peak, limit pamięci, okna, długość najgłośniejszego fragmentu, rozdzielczość FFT.
- `bands` / `band_groups` — zakresy pasm i grupy (bass/mid/high) używane w porównaniach.
- `normalization` — domyślny cel LUFS i sufit TP.
- `codec_simulation` — lista enkoderów FFmpeg i bitrate'ów.
- `thresholds` — wszystkie progi ostrzegawcze diagnozy.
- `genre_profiles` — zgrubne granice udziału energii pasm; używane **tylko bez referencji** i tylko do flagowania skrajności.

Własny plik przez `--config` jest scalany z domyślnym (wystarczy podać zmieniane klucze).

---

## 8. Obsługa błędów

Czytelne komunikaty (bez tracebacków) dla: brakującego pliku, nieobsługiwanego rozszerzenia,
uszkodzonego/nieczytelnego pliku, NaN/Inf w danych, ciszy cyfrowej, zbyt krótkiego nagrania (< 0.4 s),
> 2 kanałów, przekroczenia limitu pamięci (`analysis.max_memory_mb`, sugeruje `--start/--end`),
braku FFmpeg (M4A/AAC i kodeki), brakujących bibliotek Pythona (sprawdzane w `main.py`),
błędnego `config.json`. Uszkodzona referencja jest pomijana z ostrzeżeniem — analiza trwa dalej.
