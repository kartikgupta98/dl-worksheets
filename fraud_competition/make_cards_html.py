"""
Builds materials/technique_cards.html from materials/technique_cards.md, so the handout has one source.

    python make_cards_html.py                    # write materials/technique_cards.html
    python make_cards_html.py --fragment PATH    # also write a version without <html>/<head>/<body>
"""
import html
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "materials" / "technique_cards.md"
OUT = HERE / "materials" / "technique_cards.html"
STEP_WORDS = ["preprocessing", "features", "after the split", "model", "training", "submission"]


def inline(text):
    text = html.escape(text, quote=False)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)


def parse(md):
    doc = dict(title="", intro=[], order_lead="", order=[], order_after=[], sections=[], pitfalls=[])
    mode, section, card = "intro", None, None
    for raw in md.splitlines():
        line = raw.strip()
        if not line or line == "---":
            continue
        if line.startswith("# "):
            doc["title"] = line[2:]
        elif line.startswith("## "):
            head = line[3:]
            if head.startswith("Where"):
                mode = "order"
            elif head == "Pitfalls":
                mode = "pitfalls"
            else:
                mode, section = "cards", dict(letter=head[0], name=head[3:], cards=[])
                doc["sections"].append(section)
        elif line.startswith("### "):
            m = re.match(r"(\w\d+)\. (.+)", line[4:])
            card = dict(id=m[1], title=m[2], fields=[])
            section["cards"].append(card)
        elif mode == "intro":
            doc["intro"].append(line)
        elif mode == "order":
            m = re.match(r"\d+\. \*\*(.+?):\*\* (.+)", line)
            if m:
                doc["order"].append((m[1], m[2]))
            elif doc["order"]:
                doc["order_after"].append(line)
            else:
                doc["order_lead"] = line
        elif mode == "cards":
            m = re.match(r"\*\*(\w+):\*\* (.+)", line)
            card["fields"].append((m[1], m[2]))
        elif mode == "pitfalls":
            m = re.match(r"- \*\*(.+?)\*\* (.+)", line)
            doc["pitfalls"].append((m[1], m[2]))
    return doc


def step_of(text):
    low = text.lower()
    hits = [(low.find(word), i + 1) for i, word in enumerate(STEP_WORDS) if word in low]
    return min(hits)[1] if hits else None


def render_body(doc):
    index = []
    for s in doc["sections"]:
        items = "".join(f'<li><a href="#{c["id"]}"><span class="id">{c["id"]}</span>{inline(c["title"])}</a></li>'
                        for c in s["cards"])
        index.append(f'<a class="index-section" href="#{s["letter"]}">{s["letter"]} &middot; {inline(s["name"])}</a>'
                     f"<ul>{items}</ul>")
    jump = "".join(f'<a href="#{s["letter"]}"><span class="id">{s["letter"]}</span>{inline(s["name"])}</a>'
                   for s in doc["sections"])

    steps = "".join(f'<li><span class="step">{i}</span><strong>{inline(name)}</strong>'
                    f'<span class="desc">{inline(desc)}</span></li>'
                    for i, (name, desc) in enumerate(doc["order"], 1))

    groups = []
    for s in doc["sections"]:
        cards = []
        for c in s["cards"]:
            fields = []
            for label, text in c["fields"]:
                value = inline(text)
                if label == "Where" and step_of(text):
                    n = step_of(text)
                    value = (f'<span class="step step-sm" title="Notebook step {n}: {html.escape(doc["order"][n - 1][0])}">'
                             f"{n}</span>{value}")
                fields.append(f'<div class="field field-{label.lower()}"><dt>{label}</dt><dd>{value}</dd></div>')
            cards.append(f'<article class="card" id="{c["id"]}">'
                         f'<header class="card-head"><span class="id">{c["id"]}</span><h3>{inline(c["title"])}</h3></header>'
                         f'<dl class="fields">{"".join(fields)}</dl></article>')
        groups.append(f'<section class="group" id="{s["letter"]}">'
                      f'<h2><span class="letter">{s["letter"]}</span>{inline(s["name"])}</h2>'
                      f'<div class="cards">{"".join(cards)}</div></section>')

    pitfalls = "".join(f"<li><strong>{inline(lead)}</strong> {inline(text)}</li>" for lead, text in doc["pitfalls"])

    return f"""<div class="page">
<nav class="index" aria-label="All cards">
<a class="index-title" href="#top">Technique cards</a>
<a class="index-section" href="#where">Where changes go</a>
{"".join(index)}
<a class="index-section" href="#pitfalls">Pitfalls</a>
</nav>
<main id="top">
<header class="masthead">
<p class="eyebrow">Flagged or Fraud? &middot; Student handout</p>
<h1>{inline(doc["title"])}</h1>
<p class="lede">{inline(" ".join(doc["intro"]))}</p>
</header>
<nav class="jump" aria-label="Sections">{jump}<a href="#pitfalls">Pitfalls</a></nav>
<section class="order" id="where">
<h2>Where changes go</h2>
<p>{inline(doc["order_lead"])}</p>
<ol class="steps">{steps}</ol>
<p class="order-note">{inline(" ".join(doc["order_after"]))}</p>
</section>
{"".join(groups)}
<section class="group" id="pitfalls">
<h2>Pitfalls</h2>
<ul class="pitfalls">{pitfalls}</ul>
</section>
</main>
</div>"""


CSS = """
:root {
  --ground: #f2f5f4;
  --surface: #ffffff;
  --ink: #17212b;
  --muted: #56636f;
  --rule: #d8dfe1;
  --accent: #0d5c61;
  --accent-soft: #dcebea;
  --code-bg: #e8eeed;
  --watch: #8a4a00;
  --serif: "Source Serif 4", Georgia, "Times New Roman", serif;
  --sans: "IBM Plex Sans", "Segoe UI", system-ui, -apple-system, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ground: #0f1417;
    --surface: #172026;
    --ink: #e3e9eb;
    --muted: #98a6ae;
    --rule: #28343b;
    --accent: #6cc2bf;
    --accent-soft: #173a3b;
    --code-bg: #212c32;
    --watch: #e6b064;
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --ground: #0f1417;
  --surface: #172026;
  --ink: #e3e9eb;
  --muted: #98a6ae;
  --rule: #28343b;
  --accent: #6cc2bf;
  --accent-soft: #173a3b;
  --code-bg: #212c32;
  --watch: #e6b064;
  color-scheme: dark;
}

* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
@media (prefers-reduced-motion: reduce) { html { scroll-behavior: auto; } }
body {
  margin: 0;
  background: var(--ground);
  color: var(--ink);
  font: 400 16px/1.6 var(--sans);
  -webkit-font-smoothing: antialiased;
}
a { color: var(--accent); }
a:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; border-radius: 3px; }
code {
  font-family: var(--mono);
  font-size: 0.86em;
  background: var(--code-bg);
  padding: 0.08em 0.35em;
  border-radius: 4px;
  overflow-wrap: anywhere;
}

.page {
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: 3.5rem;
  max-width: 1080px;
  margin: 0 auto;
  padding: 3.5rem 1.25rem 6rem;
}
main { min-width: 0; }

/* index */
.index { display: none; }
.index a { text-decoration: none; }
.index-title {
  display: block;
  font: 600 0.6875rem/1.2 var(--sans);
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 1.25rem;
}
.index-section {
  display: block;
  font-weight: 600;
  font-size: 0.8125rem;
  color: var(--ink);
  margin: 1.1rem 0 0.35rem;
}
.index ul { list-style: none; margin: 0; padding: 0; display: grid; gap: 0.1rem; }
.index li a {
  display: grid;
  grid-template-columns: 1.9rem minmax(0, 1fr);
  padding: 0.12rem 0;
  font-size: 0.8125rem;
  line-height: 1.4;
  color: var(--muted);
}
.index li a:hover, .index-section:hover { color: var(--accent); }
.index .id, .jump .id { font: 600 0.72rem/1.6 var(--mono); color: var(--accent); }
@media (min-width: 1000px) {
  .page { grid-template-columns: 210px minmax(0, 760px); justify-content: center; }
  .index {
    display: block;
    position: sticky;
    top: 2rem;
    align-self: start;
    max-height: calc(100vh - 4rem);
    overflow-y: auto;
    padding-right: 0.5rem;
  }
  .jump { display: none; }
}

/* masthead */
.eyebrow {
  margin: 0 0 0.75rem;
  font: 600 0.75rem/1.2 var(--sans);
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--accent);
}
h1 {
  margin: 0 0 1rem;
  font: 600 2.625rem/1.1 var(--serif);
  letter-spacing: -0.01em;
  text-wrap: balance;
}
.lede { margin: 0; max-width: 62ch; font-size: 1.0625rem; color: var(--muted); }

.jump { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 1.75rem; }
.jump a {
  display: inline-flex;
  gap: 0.4rem;
  align-items: baseline;
  padding: 0.3rem 0.7rem;
  border: 1px solid var(--rule);
  border-radius: 999px;
  background: var(--surface);
  color: var(--ink);
  font-size: 0.8125rem;
  text-decoration: none;
}

/* notebook order */
.order {
  margin-top: 2.5rem;
  padding: 1.5rem 1.5rem 1.35rem;
  background: var(--surface);
  border: 1px solid var(--rule);
  border-radius: 10px;
  scroll-margin-top: 1.5rem;
}
.order h2 { margin: 0 0 0.25rem; font: 600 1.3125rem/1.3 var(--serif); }
.order p { margin: 0; color: var(--muted); font-size: 0.9375rem; max-width: 64ch; }
.steps {
  list-style: none;
  margin: 1.25rem 0 1.25rem;
  padding: 0;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1.1rem 1.5rem;
}
.steps li {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr);
  align-content: start;
  align-items: start;
  column-gap: 0.7rem;
}
.steps strong { font-weight: 600; line-height: 1.6rem; }
.steps .desc { grid-column: 2; font-size: 0.875rem; line-height: 1.5; color: var(--muted); }
.step {
  grid-row: span 2;
  display: inline-grid;
  place-items: center;
  width: 1.6rem;
  height: 1.6rem;
  border: 1.5px solid var(--accent);
  border-radius: 50%;
  color: var(--accent);
  font: 600 0.8125rem/1 var(--mono);
  font-variant-numeric: tabular-nums;
}
.order-note { padding-top: 1rem; border-top: 1px solid var(--rule); }

/* card groups */
.group { margin-top: 3.5rem; scroll-margin-top: 1.5rem; }
.group > h2 {
  display: flex;
  align-items: baseline;
  gap: 0.75rem;
  margin: 0 0 1.25rem;
  padding-bottom: 0.75rem;
  border-bottom: 1px solid var(--rule);
  font: 600 1.625rem/1.2 var(--serif);
  text-wrap: balance;
}
.letter { font: 600 0.875rem/1 var(--mono); color: var(--accent); letter-spacing: 0.04em; }
.cards { display: grid; gap: 1rem; }
.card {
  padding: 1.25rem 1.5rem 1.35rem;
  background: var(--surface);
  border: 1px solid var(--rule);
  border-radius: 8px;
  scroll-margin-top: 1.5rem;
}
.card:target { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }
.card-head { display: flex; align-items: baseline; gap: 0.6rem; margin-bottom: 0.9rem; }
.card-head .id { font: 600 0.8125rem/1 var(--mono); color: var(--accent); }
.card-head h3 { margin: 0; font: 600 1.0625rem/1.35 var(--sans); text-wrap: balance; }
.fields { display: grid; gap: 0.6rem; margin: 0; }
.field { display: grid; grid-template-columns: 4.25rem minmax(0, 1fr); gap: 1rem; }
.field dt {
  font: 600 0.6875rem/1.5rem var(--sans);
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
}
.field dd { margin: 0; max-width: 64ch; font-size: 0.9375rem; line-height: 1.6; }
.field-watch dt { color: var(--watch); }
.step-sm {
  width: 1.3rem;
  height: 1.3rem;
  margin-right: 0.45rem;
  border-width: 1.25px;
  font-size: 0.6875rem;
  vertical-align: 0.05em;
}
@media (max-width: 560px) {
  h1 { font-size: 2.125rem; }
  .card { padding: 1.1rem 1.1rem 1.2rem; }
  .field { grid-template-columns: minmax(0, 1fr); gap: 0; }
}

/* pitfalls */
.pitfalls { list-style: none; margin: 0; padding: 0; }
.pitfalls li { padding: 0.8rem 0; border-bottom: 1px solid var(--rule); max-width: 66ch; font-size: 0.9375rem; }
.pitfalls li:first-child { padding-top: 0.2rem; }
.pitfalls strong { font-weight: 600; }

@media print {
  body { background: #fff; color: #000; font-size: 11pt; }
  .index, .jump { display: none; }
  .page { display: block; padding: 0; max-width: none; }
  .card, .order, .steps li { break-inside: avoid; }
  .card, .order { border-color: #bbb; }
  .group { margin-top: 1.5rem; }
}
"""

HEAD = f"""<title>Flagged or Fraud? Technique Cards</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600&family=Source+Serif+4:opsz,wght@8..60,600&display=swap">
<style>{CSS}</style>"""


if __name__ == "__main__":
    body = render_body(parse(SRC.read_text()))
    OUT.write_text(f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
{HEAD}
</head>
<body>
{body}
</body>
</html>
""")
    print("wrote", OUT)
    if "--fragment" in sys.argv:
        path = Path(sys.argv[sys.argv.index("--fragment") + 1])
        path.write_text(f"{HEAD}\n{body}\n")
        print("wrote", path)
