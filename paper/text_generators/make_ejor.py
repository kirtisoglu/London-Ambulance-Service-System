"""Build the EJOR (elsarticle) version of the paper from main_arxiv.tex.

Writes src/main_ejor.tex and src/highlights.tex, then compiles main_ejor.pdf.
The arXiv file stays the single source of the text; this script only swaps the
preamble, the front matter and the bibliography style.
"""
import pathlib, re, subprocess, sys

SRC = pathlib.Path("src"); s = (SRC / "main_arxiv.tex").read_text()

def rep(old, new, cnt=1):
    global s
    pat = re.escape(old).replace('\\ ', r'\s+'); n = len(re.findall(pat, s)); assert n == cnt, (old[:70], n)
    s = re.sub(pat, lambda m: new, s)

rep(r"\documentclass[11pt]{article}", r"\documentclass[preprint,12pt,authoryear]{elsarticle}")
rep(r"\usepackage[margin=1in]{geometry}", "")
rep(r"\usepackage[numbers,sort&compress]{natbib}", "")
rep(r"\usepackage[mathlines]{lineno}", "")
rep(r"\linenumbers", "")
# front matter
start = s.index(r"\title{FalCom"); end = s.index(r"\end{abstract}") + len(r"\end{abstract}")
abstract = s[s.index(r"\begin{abstract}") + len(r"\begin{abstract}"): s.index(r"\end{abstract}")]
front = r"""\journal{European Journal of Operational Research}

\begin{frontmatter}

\title{FalCom: An MCMC Sampling Framework for Facility Location and Districting Problems}

\author[iit]{Alaittin K\i rt\i \c{s}o\u{g}lu\corref{cor1}}
\ead{kirtisoglu@hawk.iit.edu}
\author[iit]{Hemanshu Kaul}
\ead{kaul@iit.edu}
\cortext[cor1]{Corresponding author.}
\affiliation[iit]{organization={Department of Applied Mathematics, Illinois Institute of Technology},
  city={Chicago}, postcode={IL 60616}, country={USA}}

\begin{abstract}""" + abstract + r"""\end{abstract}

\begin{keyword}
Location \sep Districting \sep Markov chain Monte Carlo \sep Hierarchical facility location \sep Emergency medical services
\end{keyword}

\end{frontmatter}"""
s = s[:start] + front + s[end:]
rep(r"\bibliographystyle{plainnat}", r"\bibliographystyle{elsarticle-harv}")
(SRC / "main_ejor.tex").write_text(s)

highlights = [
    "An MCMC sampler for two-level districting with capacitated facility assignment",
    "Every proposal is a feasible plan: contiguous, balanced and anchored at a facility",
    "Runs on real candidate sets as sparse as London's 66 ambulance stations",
    "Validated by exact enumeration, multi-start agreement and proven MILP optima",
    "A London ensemble bounds the station-leg access cost of workload balance",
]
for h in highlights:
    assert len(h) <= 85, (len(h), h)
(SRC / "highlights.tex").write_text(
    "\\documentclass{elsarticle}\n\\begin{document}\n\\begin{frontmatter}\n\\title{Highlights}\n\\end{frontmatter}\n"
    "\\begin{itemize}\n" + "".join(f"\\item {h}\n" for h in highlights) + "\\end{itemize}\n\\end{document}\n")
print("wrote main_ejor.tex and highlights.tex")
