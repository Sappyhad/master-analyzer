Optional reference tracks. Every supported audio file in this folder is analyzed automatically
and compared with the input (unless you pass --no-reference).

Tips:
  * Use 2-5 commercial masters of the same genre and similar arrangement.
  * Prefer lossless files. Lossy references (MP3/AAC/streaming rips) carry codec artefacts:
    their true peak and top octave are not the master's.
  * No single reference is a quality target; MasterCheck reports per-reference results and the median.
  * Compare matching sections with --ref-start / --ref-end (applied to every reference).

If this folder is empty, MasterCheck analyzes the master alone without errors.
