# Build configuration for the weekly progress reports.
#
# Class file (ieeeconf.cls), bibliography style (IEEEtran.bst) and shared
# preamble (usepackage.tex, IEEEabrv.bib) all live in ./texmf/. We extend
# TEXINPUTS / BIBINPUTS / BSTINPUTS so latex, bibtex and friends can find
# them while keeping the top-level folder clean.
ensure_path('TEXINPUTS', './texmf//');
ensure_path('BIBINPUTS', './texmf//');
ensure_path('BSTINPUTS', './texmf//');

# Default to building the latest week. Override on the CLI:
#   latexmk -pdf week3
@default_files = ('week4.tex');
$pdf_mode = 1;

