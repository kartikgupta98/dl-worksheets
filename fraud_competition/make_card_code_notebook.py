"""
Builds materials/instructor_card_code.ipynb: for every technique card, the code that implements
it, to refer to while helping students. Instructor only.

    python make_card_code_notebook.py

The code is the card code in make_notebooks.py, the same code test_cards.py runs. Each card is
shown in the context it was tested in: on top of the cards before it in the instructor guide's
path, or on top of the full solution for cards off that path.
"""
import json
from difflib import SequenceMatcher
from pathlib import Path

from make_cards_html import SRC, parse
from make_notebooks import EXTRA, FULL, LADDER, build, cards_up_to, code, md

HERE = Path(__file__).parent
OUT = HERE / "materials" / "instructor_card_code.ipynb"
PARTIAL = {"Imports", "Preprocessing"}  # show only the new lines of these cells


def contexts():
    ctx, done = {}, []
    for _, cards in LADDER:
        for card in cards:
            ctx[card] = list(done)
            done.append(card)
    for _, base, cards in EXTRA:
        ctx.setdefault(cards[0], cards_up_to(base))
    return ctx


def sections(nb):
    heading, out = "", []
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if cell["cell_type"] == "markdown":
            heading = src[3:] if src.startswith("## ") else "Imports"
        else:
            out.append((heading, src.splitlines()))
    return out


def snippets(card, before):
    cells = []
    for (name, a), (_, b) in zip(sections(build(before)), sections(build(before + [card]))):
        if a == b:
            continue
        lines = []
        for op, i1, i2, j1, j2 in SequenceMatcher(None, a, b).get_opcodes():
            if op == "equal":
                if name not in PARTIAL:
                    lines += b[j1:j2]
                continue
            lines += [f"{line[:len(line) - len(line.lstrip())]}# removed: {line.strip()}"
                      for line in a[i1:i2] if line.strip()]
            lines += [f"{line}  # {card}" if line.strip() else line for line in b[j1:j2]]
        if name == "Preprocessing":
            lines = ["for df in [train, test]:", "    ...  # your earlier preprocessing lines"] + lines
        cells.append(code([f"# Section: {name}"] + lines))
    return cells


def builds_on(context):
    if context == cards_up_to(FULL):
        return "the full solution (every card in the guide's path)"
    return ", ".join(context) if context else "the starter notebook"


if __name__ == "__main__":
    doc, ctx = parse(SRC.read_text()), contexts()
    cells = [md("# Technique card code (instructor reference)\n\n"
                "The code for every technique card, to refer to while helping students. Not meant to be run top "
                "to bottom.\n\n"
                "- Each code cell starts with the notebook section it belongs to.\n"
                "- Lines ending in the card ID (e.g. `# A6`) are new or changed; `# removed:` lines are deleted.\n"
                "- Preprocessing cells show only the new lines inside the `for df in [train, test]:` loop; other "
                "sections show the whole cell.\n"
                "- *Builds on* lists the cards already applied when this code was tested (`test_cards.py`). A "
                "student's notebook may differ; the idea matters, not an exact match.")]
    for section in doc["sections"]:
        cells.append(md(f"# {section['letter']}. {section['name']}"))
        for card in section["cards"]:
            fields = dict(card["fields"])
            cells.append(md(f"## {card['id']}. {card['title']}\n\n"
                            f"**Where:** {fields['Where']}  \n"
                            f"*Builds on:* {builds_on(ctx[card['id']])}"))
            cells += snippets(card["id"], ctx[card["id"]])
    notebook = {"cells": cells, "nbformat": 4, "nbformat_minor": 4,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}}}
    OUT.write_text(json.dumps(notebook, indent=1))
    print("wrote", OUT)
