# Build configuration for the topic introduction + literature review document.
#
# Class file (ieeeconf.cls), bibliography style (IEEEtran.bst) and the
# IEEEabrv abbreviation bib live in ./texmf/. Extend TEXINPUTS / BIBINPUTS /
# BSTINPUTS so latex and bibtex can find them.
ensure_path('TEXINPUTS', './texmf//');
ensure_path('BIBINPUTS', './texmf//');
ensure_path('BSTINPUTS', './texmf//');

@default_files = ('topic_introduction_and_literature_review.tex');
$pdf_mode = 1;
