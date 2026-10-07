# Pre-Phase-D K=0 result (BUGGY)

This is the K=0 sweep that hit **0 / 50 = 0.0%** test success
under the chunk-boundary `env.reset()` controller-state bug
(`unified_loop.py:466-468` pre-Phase-D, `--n-workers 1` default).

Root cause and fix: see
`../../phase_d_2026_05_30/BRIEF.md`.

Files preserved here for traceability against the post-fix run
in the sibling `K0/` directory.
