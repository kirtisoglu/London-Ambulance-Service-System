"""
Restructure the experiments of ``main_arxiv.tex``.

Starts from the committed manuscript (``git show HEAD:paper/sources/main_arxiv.tex``),
so it can be re-run at any time, and

* replaces Section 7 (old lines "Computational Experiments" .. end of the London
  case study) by ``\\input{tex/sec_experiments}`` and ``\\input{tex/sec_london}``;
* moves the detailed London diagnostics (convergence paragraph, holding times,
  fifth chain, the two replications) verbatim into a new appendix,
  ``tex/app_london.tex``;
* updates the three sentences outside the section that describe it (abstract,
  roadmap, the forward reference to the holding-time measurement).

The tables and figures the new sections input are written by
``make_sec7_grids.py`` and ``make_sec8_london.py``.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "paper/sources"
BASE = "9623b6d"      # the commit whose Section 7 this script replaces

# This transformation was applied once (commit d3a6c74). The manuscript has been edited by hand since
# (Appendix C removed, the scaling figure dropped), so re-running it would discard those edits.
import sys
if "--force" not in sys.argv:
    raise SystemExit("apply_sec7_restructure.py was applied at d3a6c74 and rebuilds main_arxiv.tex from "
                     f"{BASE}; later manual edits would be lost. Pass --force to run it anyway.")

orig = subprocess.run(["git", "show", f"{BASE}:paper/sources/main_arxiv.tex"], cwd=REPO, capture_output=True,
                      text=True, check=True).stdout
lines = orig.split("\n")


def find(prefix, start=0):
    for i in range(start, len(lines)):
        if lines[i].startswith(prefix):
            return i
    raise SystemExit(f"anchor not found: {prefix}")


i_sec = find(r"\section{Computational Experiments}")
i_disc = find(r"\section{Discussion and Conclusions}")
i_start = i_sec - 1                       # the comment rule above the section
i_end = i_disc - 1                        # the comment rule above the discussion
old = lines[i_start:i_end]


def para(prefix):
    hits = [l for l in old if l.startswith(prefix)]
    if len(hits) != 1:
        raise SystemExit(f"paragraph not found uniquely: {prefix!r} ({len(hits)})")
    return hits[0]


# ---- appendix with the detailed London diagnostics, verbatim -----------------
app = [
    r"\section{Additional diagnostics of the London chains}",
    r"\label{app:london-extra}",
    "",
    r"This appendix collects the detailed diagnostics behind Sections~\ref{sec:exp-multistart} and~\ref{sec:exp-emergency}. "
    r"Here the \emph{energy} is the total $E = E_1 + E_2$ of the station leg and the Group leg, and "
    r"Kolmogorov--Smirnov distances are on every twentieth step. Steps to forget are counted here against the plateau of "
    r"the chain's own curve (the first step within $0.02$ of it), whereas Table~\ref{tab:multistart} counts them against "
    r"the share that two independent plans have in common; the two definitions give $320$ to $520$ and $400$ to $520$ steps "
    r"for the four sector-start chains, and $480$ and $680$ for the fifth chain.",
    "",
    r"\paragraph{Agreement of the four chains.}",
    para("The four chains agree after burn-in.").replace("and the frequencies below are quoted", "and the frequencies of Section~\\ref{sec:las-structure} are quoted")
    .replace("The agreement extends to the readouts reported below,", "The agreement extends to the readouts of Section~\\ref{sec:las-structure},"),
    "",
    r"\paragraph{Holding times and the law of the accepted moves.}",
    para("Holding the state on rejection reweights the sampled law"),
    "",
    r"\paragraph{A chain from a structurally different start.}",
    para("The four chains share the sector-wise construction"),
    "",
    r"\paragraph{Replication under a uniform demand proxy.}",
    para("Nor does it depend on the within-borough apportionment").replace(
        "Nor does it depend on the within-borough apportionment, as far as a second proxy can tell. We repeat",
        "We repeat"),
    "",
    r"\paragraph{Replication at tolerance $0.25$.}",
    para("The result does not depend on the tolerance.").replace("The result does not depend on the tolerance. ", ""),
    "",
    r"\paragraph{Station and Group pairs in full.}",
    r"Stations open in more than $90\%$ of plans are led by Fulham ($97\%$), St John's Wood ($96\%$), Ilford, Deptford and "
    r"Barnehurst ($94\%$ each) and Whipps Cross ($93\%$); the least used are Chase Farm ($40\%$), Heathrow Airport ($41\%$), "
    r"Twickenham ($49\%$), New Addington and Wembley ($60\%$), and Tolworth, Barnet and Richmond ($61\%$). Inner stations "
    r"such as St John's Wood, Ilford, Camden, Deptford and Brent carry three units in $37\%$ to $42\%$ of the plans in which "
    r"they are open, and outer stations such as Tolworth, Richmond, Chase Farm, Heathrow Airport and New Addington carry one "
    r"unit in $67\%$ to $77\%$ of theirs. The pairs kept together most often inside a real Group are Hornchurch and Romford "
    r"($67\%$), Barnehurst and Woolwich ($55\%$), Becontree and Hornchurch ($52\%$), and Oval and Waterloo and Ponders End and "
    r"Chase Farm ($49\%$ each); the strongest pairs that the real Groups separate are Barnehurst and St Pauls Cray ($49\%$, "
    r"Greenwich and Bromley Groups), Richmond and Putney ($48\%$), Fulham and Westminster ($47\%$), Waterloo and Deptford "
    r"($42\%$) and Heathrow Airport and Feltham ($41\%$).",
    "",
]
(SRC / "tex/app_london.tex").write_text("\n".join(app))

# ---- main file ------------------------------------------------------------------
new = lines[:i_start] + [r"\input{tex/sec_experiments}", "", r"\input{tex/sec_london}", "", r"\FloatBarrier", ""] + lines[i_end:]
text = "\n".join(new)


def rep(a, b, count=1):
    global text
    if text.count(a) != count:
        raise SystemExit(f"expected {count} occurrence(s) of {a[:60]!r}, found {text.count(a)}")
    text = text.replace(a, b)


rep(r"\end{document}", "\\input{tex/app_london}\n\n\\end{document}")

# abstract: the annealing sentence becomes the placement of the ensemble
a0 = text.index("Run as an annealing optimizer on grids of $100$ and $400$ basic units")
a1 = text.index("Finally, we apply FalCom to the London Ambulance Service")
text = (text[:a0]
        + "Ablations show which components of the proposal the chain cannot run without. On grids of $100$ and $400$ "
          "basic units a typical sampled plan lies $40$ to $50\\%$ above the proven optimum of a median objective that "
          "the sampler never sees, and the same proposal run as an annealer comes within $1\\%$ and $8\\%$ of it. \n    "
        + text[a1:])

# roadmap (whitespace-tolerant: the source lines carry trailing blanks)
import re
pat = re.compile(r"Section~\\ref\{sec:experiments\}\s+validates the sampler against exact enumeration.*?Ambulance Service\.\s+Finally, ",
                 re.S)
if len(pat.findall(text)) != 1:
    raise SystemExit("roadmap sentence not found")
text = pat.sub(lambda m: (
    "Section~\\ref{sec:experiments} validates the sampler against exact enumeration and across independently "
    "started chains, ablates its components, evaluates its scaling on large synthetic grids and places the "
    "ensemble relative to exact optima of the mixed-integer model. Section~\\ref{sec:exp-emergency} applies it "
    "to the London Ambulance Service. Finally, "), text)

# the holding-time measurement now lives in the appendix
rep("Section~\\ref{sec:exp-emergency} measures the size of that effect on the London readouts.",
    "Appendix~\\ref{app:london-extra} measures the size of that effect on the London readouts.")

(SRC / "main_arxiv.tex").write_text(text)
print(f"replaced lines {i_start + 1}-{i_end} of the committed manuscript ({i_end - i_start} lines); "
      f"new file has {text.count(chr(10)) + 1} lines")
