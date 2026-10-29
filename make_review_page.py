"""
Build the coauthor review page: every figure and table of the TMLR submission, in paper order,
with only its numbered title (no body text, no captions) and one Model selector for all of them.

    python make_review_page.py OUT.html

Read from submission.md, so the page always matches the paper. The Pages workflow writes it to
review/index.html next to the paper, and the figure files it embeds are the paper's own.
"""
import html
import re
import sys

SUBMISSION = 'tmlr-beyond-pdf-author-kit/submission_folder/submission.md'
ASSET = re.compile(r"\{\{\s*'([^']+)'\s*\|\s*relative_url\s*\}\}")


def title_of(fragment):
    """'Figure 4: Semantic projection …' from a caption's bold lead, or 'Table 3: <first sentence>'."""
    m = re.search(r'<b>((?:Figure|Table) \d+:)\s*([^<]*)</b>\s*([^<]*)', fragment)
    if not m:
        return None
    label, bold, rest = m.groups()
    text = bold.strip() or re.split(r'(?<=\.)\s', rest.strip(), maxsplit=1)[0]
    return f'{label} {text}'.strip()


def markdown_table(lines):
    """A GitHub-style pipe table as HTML (bold and <br> kept, nothing else needed here)."""
    cells = [[c.strip() for c in line.strip().strip('|').split('|')] for line in lines]
    align = ['center' if a.startswith(':') and a.endswith(':') else 'right' if a.endswith(':') else 'left'
             for a in cells[1]]
    inline = lambda c: re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', c)
    head = ''.join(f'<th style="text-align:{a}">{inline(c)}</th>' for c, a in zip(cells[0], align))
    body = ''.join('<tr>' + ''.join(f'<td style="text-align:{a}">{inline(c)}</td>' for c, a in zip(row, align))
                   + '</tr>' for row in cells[2:])
    return f'<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def items(md):
    """(position, title, html) for each figure block and each captioned markdown table."""
    out = []
    for m in re.finditer(r'<figure\b.*?</figure>', md, re.S):
        block = m.group(0)
        title = title_of(block)
        media = re.findall(r'<(?:iframe|img)\b[^>]*>(?:</iframe>)?', block)
        if title and media:
            media = [ASSET.sub(lambda a: '../' + a.group(1), tag) for tag in media]
            out.append((m.start(), title, '\n'.join(media)))
    for m in re.finditer(r'<p[^>]*>(<b>Table \d+:</b>.*?)</p>\s*\n\s*\n((?:\|.*\|\s*\n)+)', md):
        out.append((m.start(), title_of(m.group(1)), markdown_table(m.group(2).strip().splitlines())))
    return sorted(out)


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex">
<title>Figures and Tables Review</title>
<link rel="icon" href="../assets/img/tmlr-favicon.ico">
<style>
  :root {{ --text: #1a1a1a; --muted: #555; --rule: #e3e3e3; --bg: #fff; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--bg); color: var(--text); font: 15px/1.45 Arial, sans-serif; }}
  header {{ position: sticky; top: 0; z-index: 10; background: rgba(255,255,255,0.96);
           border-bottom: 1px solid var(--rule); }}
  .bar {{ max-width: 960px; margin: 0 auto; padding: 10px 16px; display: flex; flex-wrap: wrap;
          gap: 8px 16px; align-items: center; justify-content: space-between; }}
  .bar h1 {{ font-size: 16px; margin: 0; font-weight: 600; }}
  .bar a {{ color: var(--muted); font-size: 13px; }}
  .bar label {{ font-size: 13px; }}
  select {{ font: inherit; font-size: 13px; padding: 3px 6px; border: 1px solid #ccc; border-radius: 4px;
            background: #fff; max-width: 70vw; }}
  main {{ max-width: 960px; margin: 0 auto; padding: 8px 16px 48px; }}
  section {{ padding: 20px 0 8px; border-bottom: 1px solid var(--rule); }}
  section:last-child {{ border-bottom: none; }}
  h2 {{ font-size: 15px; font-weight: 600; margin: 0 0 10px; }}
  iframe {{ display: block; width: 100%; border: none; }}
  img {{ display: block; max-width: 100%; height: auto; margin: 0 auto; }}
  .scroll {{ overflow-x: auto; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th, td {{ border-bottom: 1px solid var(--rule); padding: 6px 8px; vertical-align: top; }}
  th {{ border-bottom: 1px solid #bbb; font-weight: 600; }}
</style>
</head>
<body>
<header><div class="bar">
  <h1>Figures and tables for review</h1>
  <label id="picker" hidden>Model <select id="model"></select></label>
  <a href="../under_review/submission/">Full paper</a>
</div></header>
<main>
{sections}
</main>
<script>
// One Model selector for every per-model figure, same protocol as the paper page: each figure
// posts {{type: 'tmlr-model-ready', labels}} on load and follows {{type: 'tmlr-model', model}}.
(function () {{
  var picker = document.getElementById('picker'), select = document.getElementById('model');
  function tell(win) {{ if (win && select.value) win.postMessage({{type: 'tmlr-model', model: select.value}}, '*'); }}
  window.addEventListener('message', function (e) {{
    if (!e.data || e.data.type !== 'tmlr-model-ready') return;
    if (!select.options.length) e.data.labels.forEach(function (l) {{ select.add(new Option(l, l.split(' · ')[0])); }});
    picker.hidden = false;
    tell(e.source);
  }});
  select.addEventListener('change', function () {{
    document.querySelectorAll('iframe').forEach(function (f) {{ tell(f.contentWindow); }});
  }});
}})();
// Each iframe takes its content's height: no inner scrollbars, no slack.
(function () {{
  function fit(f) {{
    try {{
      var body = f.contentDocument && f.contentDocument.body;
      if (!body || !body.offsetHeight) return;
      var cs = f.contentWindow.getComputedStyle(body);
      f.style.height = Math.ceil(body.offsetHeight + parseFloat(cs.marginTop) + parseFloat(cs.marginBottom)) + 'px';
    }} catch (e) {{}}
  }}
  function watch(f) {{
    fit(f);
    try {{ new ResizeObserver(function () {{ fit(f); }}).observe(f.contentDocument.body); }} catch (e) {{}}
  }}
  document.querySelectorAll('iframe').forEach(function (f) {{
    f.addEventListener('load', function () {{ watch(f); }});
    watch(f);
  }});
}})();
</script>
</body>
</html>
"""


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'review.html'
    md = open(SUBMISSION, encoding='utf-8').read()
    sections = []
    for _, title, body in items(md):
        wrap = f'<div class="scroll">{body}</div>' if body.startswith('<table') else body
        sections.append(f'<section>\n<h2>{html.escape(title, quote=False)}</h2>\n{wrap}\n</section>')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(PAGE.format(sections='\n'.join(sections)))
    print(f'wrote {out}: {len(sections)} figures and tables')


if __name__ == '__main__':
    main()
