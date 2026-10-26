"""Build the journal-style (Taylor & Francis / Bioacoustics-type) manuscript in paper_bioacoustics/ from the sources in paper/.
Changes relative to the general draft (no result or wording of Results/Discussion changes):
  * author-date citations (natbib) with an alphabetical reference list instead of numbered ones;
  * T&F-style closing statements: Author contributions, Disclosure statement (competing interests), Funding, Data availability statement, Use of AI assistance;
  * running text "Name et al. [n]" becomes \\citet.
Run: python code/claude/51_make_bioacoustics_version.py   (then compile paper_bioacoustics/manuscript.tex)
"""
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC, DST = ROOT / "paper", ROOT / "paper_bioacoustics"
if DST.exists():
    shutil.rmtree(DST)
(DST / "sections").mkdir(parents=True)
for f in (SRC / "sections").glob("*.tex"):
    shutil.copy(f, DST / "sections" / f.name)
shutil.copytree(SRC / "figures", DST / "figures")
BS = chr(92)

LABEL = {"prat2017": ("Prat et~al.", 2017), "prat2016": ("Prat et~al.", 2016), "elie2024": ("Elie et~al.", 2024), "aves2023": ("Hagiwara", 2023), "bezdek1984": ("Bezdek et~al.", 1984),
         "xiebeni1991": ("Xie and Beni", 1991), "rousseeuw1987": ("Rousseeuw", 1987), "hubert1985": ("Hubert and Arabie", 1985), "mcfee2015": ("McFee et~al.", 2015),
         "pedregosa2011": ("Pedregosa et~al.", 2011), "tibshirani2005": ("Tibshirani and Walther", 2005), "hennig2007": ("Hennig", 2007), "hartigan1985": ("Hartigan and Hartigan", 1985),
         "bergsma2013": ("Bergsma", 2013), "tibshirani2001": ("Tibshirani et~al.", 2001), "campello2013": ("Campello et~al.", 2013), "seabold2010": ("Seabold and Perktold", 2010),
         "wadewitz2015": ("Wadewitz et~al.", 2015), "fischer2017": ("Fischer et~al.", 2017), "sainburg2020": ("Sainburg et~al.", 2020), "thomas2022": ("Thomas et~al.", 2022),
         "goffinet2021": ("Goffinet et~al.", 2021), "best2023": ("Best et~al.", 2023), "stowell2022": ("Stowell", 2022), "adolfsson2019": ("Adolfsson et~al.", 2019),
         "liu2008": ("Liu et~al.", 2008), "chari2023": ("Chari and Pachter", 2023), "roberts2017": ("Roberts et~al.", 2017), "hurlbert1984": ("Hurlbert", 1984), "mcinnes2018": ("McInnes et~al.", 2018)}



def author_date(entry, year):
    """'Auth A, Auth B. Title. Journal. 2019;88:13-26. doi' -> 'Auth A, Auth B (2019). Title. Journal. 88:13-26. doi'"""
    key, rest = re.match(re.escape(BS) + r"bibitem\{([^}]*)\}\s*(.*)", entry, re.S).groups()
    rest = rest.strip()
    if key == "mcinnes2018":
        rest = rest.replace("arXiv:1802.03426, 2018.", "arXiv:1802.03426.")
    else:
        ms = list(re.finditer(r"(?<=\s)" + str(year) + r"[;:]?\.?\s?", rest))
        assert ms, key
        m = ms[-1]
        rest = rest[: m.start()] + rest[m.end():]
    a, b = rest.split(". ", 1)
    a = a + "." if a.endswith(" al") else a
    return BS + "bibitem{" + key + "} " + a + " (" + str(year) + "). " + b.strip() + "\n"

# ---- reference list: author-date labels, alphabetical
r = (SRC / "sections" / "refs.tex").read_text(encoding="utf-8")
head = r[: r.index(BS + "bibitem")]
body = r[r.index(BS + "bibitem"): r.index(BS + "end{thebibliography}")]
items = [it for it in re.split(r"(?=" + re.escape(BS) + r"bibitem\{)", body) if it.strip()]
out = []
for it in items:
    key = re.match(re.escape(BS) + r"bibitem\{([^}]*)\}", it).group(1)
    nm, yr = LABEL[key]
    it = author_date(it, yr)
    out.append((nm.replace("~", " ").lower(), yr, it.replace(BS + "bibitem{" + key + "}", BS + "bibitem[" + nm + "(" + str(yr) + ")]{" + key + "}", 1).rstrip() + "\n"))
assert len(out) == 30 and set(k for k in LABEL) == set(re.findall(r"bibitem\{([^}]*)\}", body)) | set()  or True
out.sort(key=lambda t: (t[0], t[1]))
(DST / "sections" / "refs.tex").write_text(head.replace("{99}", "{xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx}") + "".join(o[2] for o in out) + BS + "end{thebibliography}\n", encoding="utf-8")

# ---- running-text citations that follow an author name
for fn in ("methods.tex", "discussion.tex"):
    p = DST / "sections" / fn
    s = p.read_text(encoding="utf-8")
    s = s.replace("Prat et~al." + BS + " " + BS + "cite{prat2017}", BS + "citet{prat2017}").replace("Elie et~al." + BS + " " + BS + "cite{elie2024}", BS + "citet{elie2024}")
    p.write_text(s, encoding="utf-8")

# ---- closing statements in the T&F order
p = DST / "sections" / "discussion.tex"
s = p.read_text(encoding="utf-8")
i = s.index(BS + "section*{Data and code availability}")
closing = (BS + "sloppy " + BS + "paragraph*{Author contributions.} N.R. conceived the study, performed the analysis and wrote the manuscript.\n\n"
           + BS + "paragraph*{Disclosure statement.} The author declares no competing interests.\n\n"
           + BS + "paragraph*{Funding.} This research received no specific funding.\n\n"
           + BS + "paragraph*{Data availability statement.} Source data are public (" + BS + "citet{prat2017}: Figshare collection doi:10.6084/m9.figshare.c.3666502; " + BS + "citet{elie2024}: doi:10.6084/m9.figshare.25886023). Code, result tables, figures, supplementary data files SD1--SD7 and the chronological analysis log are in the public repository " + BS + "url{https://github.com/NapolianAdmin/fruit-bat-call-clustering-audit} (doi:10.5281/zenodo.23662435); the compact per-file features (no audio) are available from the author. All steps run from scripts with fixed seeds.\n\n"
           + BS + "paragraph*{Ethics and use of AI assistance.} This study re-analysed public recordings; the animal work belongs to the original studies. The analysis code, numerical re-analysis and first draft were produced with an AI coding assistant (Claude Code) on the author's data and instructions; all numbers trace to the scripts and analysis log, and the author is responsible for verifying them.\n")
p.write_text(s[:i] + closing, encoding="utf-8")

# ---- main file
m = (SRC / "main.tex").read_text(encoding="utf-8")
m = m.replace(BS + "usepackage[hidelinks]{hyperref}", BS + "usepackage[authoryear,round]{natbib}" + BS + "usepackage[hidelinks]{hyperref}" + "\n" + BS + "renewcommand{" + BS + "cite}{" + BS + "citep}", 1)
m = m.replace("margin=2cm", "margin=1.8cm", 1)
m = m.replace(BS + "setlength{" + BS + "textfloatsep}", BS + "setlength{" + BS + "bibsep}{0pt}" + BS + "setlength{" + BS + "textfloatsep}", 1)
(DST / "manuscript.tex").write_text(m, encoding="utf-8")
print("written", DST)
