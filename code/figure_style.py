"""Shared Plotly style for the TMLR figures (74, 75, 79, 82 and their IT companions 83, 85, 86).

Type follows 79: Arial, 13 pt base, 12 pt ticks / annotations / table cells, 14 pt axis titles.
The green is 74's operating-peak green (Okabe-Ito bluish green), also 75's rank-1 fill.

Exports go to the TMLR submission at the root of this repo, or wherever TMLR_REPO points.
"""
import os

TMLR_REPO = os.path.abspath(os.environ.get("TMLR_REPO", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
TMLR_DIR = os.path.join(TMLR_REPO, "tmlr-beyond-pdf-author-kit", "submission_folder", "assets", "html", "submission")

FAMILY = "Arial, sans-serif"
FONT = dict(family=FAMILY, size=13, color="black")   # base text: legends, panel titles, cell values
TICK = dict(family=FAMILY, size=12, color="black")   # tick labels, in-plot annotations, table cells
TITLE = dict(family=FAMILY, size=14, color="black")  # axis titles, column titles

GREEN = "#009E73"
# 75's rank-matrix tints of GREEN (rank 1 / 2 / 3), light -> dark.
GREEN_TINTS = ["#D1EEE6", "#80CEB9", GREEN]


def require_tmlr_repo():
    """Fail before writing when the public repo is not cloned, rather than leave an orphan copy."""
    if not os.path.isdir(os.path.join(TMLR_REPO, ".git")):
        raise FileNotFoundError(f"TMLR submission repo not found at {os.path.abspath(TMLR_REPO)}: clone "
                                "PrecisionNeuroLab/Symptom-Vectors-for-Depression there or set TMLR_REPO")


def write_tmlr_html(fig, path):
    """Write a figure for the TMLR Beyond-PDF page: full container width and responsive, so it
    reflows on a phone instead of overflowing at a fixed pixel width."""
    require_tmlr_repo()
    fig.write_html(path, include_plotlyjs="cdn", full_html=True, default_width="100%",
                   config={"responsive": True, "displaylogo": False})


SELECT_STYLE = (f"font-family: {FAMILY}; font-size: 12px; color: black; padding: 3px 6px; "
                "border: 1px solid #ccc; border-radius: 3px; background: white; max-width: 100%;")


# Embedded figures follow the page's model choice (submission.md's section dropdowns), by name:
# a button label is "<model> · L<layer>", and the page sends {type: MODEL_MESSAGE, model: <model>}.
# On load an embedded figure posts {type: READY_MESSAGE, labels: [...]} so the page can fill its
# dropdowns and reply with the current choice.
MODEL_MESSAGE, READY_MESSAGE = "tmlr-model", "tmlr-model-ready"
DROPDOWN_JS = """
const select = document.getElementById('view');
const status = document.getElementById('view-status');
const LABELS = Array.from(select.options, o => o.text);
const embedded = window.parent !== window;
let drawing = 0;  // latest requested view: an earlier redraw finishing late must not relabel
function settle(k) {  // label the view actually on screen
  status.textContent = embedded ? 'Model: ' + LABELS[k] : '';
  status.style.color = '#555';
}
function show(k) {
  if (k < 0 || k === +select.value) return;
  select.value = k;
  const ticket = ++drawing;
  status.textContent = 'Updating to ' + LABELS[k] + '…';
  status.style.color = '#999';
  const [data, layout] = VIEWS[k];
  // Not requestAnimationFrame: browsers pause it in off-screen iframes, so the label would stick.
  Plotly.update('figure', data, layout).then(() => { if (ticket === drawing) settle(k); });
}
select.addEventListener('change', e => show(+e.target.value));
settle(0);
if (embedded) {  // the page's Model selector drives this figure; the status line names its model
  select.style.display = 'none';
  document.getElementById('view-label').style.display = 'none';
  window.addEventListener('message', e => {
    if (e.data && e.data.type === '%s') show(LABELS.findIndex(l => l.split(' · ')[0] === e.data.model));
  });
  window.parent.postMessage({type: '%s', labels: LABELS}, '*');
}
""" % (MODEL_MESSAGE, READY_MESSAGE)


def write_tmlr_dropdown_html(fig, path, label="Model"):
    """write_tmlr_html for a figure whose first updatemenu switches between views (one button per
    model): the menu is replaced by a plain HTML <select> above the chart that runs the same
    Plotly.update calls. Embedded in the page, the select is hidden and the figure follows the
    page's Model selector instead, under a status line naming the model it shows; the line reads
    "Updating to ..." until the redraw has finished (DROPDOWN_JS). Nothing sits in the plot area, so no figure
    reserves room for a menu (and Plotly tables, which drop header cells when offset from the
    top, render whole)."""
    require_tmlr_repo()
    import plotly.graph_objects as go
    from plotly.io.json import to_json_plotly

    buttons = fig.layout.updatemenus[0].buttons
    views = to_json_plotly([[b.args[0], b.args[1] if len(b.args) > 1 else {}] for b in buttons])
    options = "".join(f'<option value="{k}">{b.label}</option>' for k, b in enumerate(buttons))
    chart = go.Figure(fig)
    chart.layout.updatemenus = ()  # update_layout(updatemenus=[]) leaves the menu in place
    html = chart.to_html(include_plotlyjs="cdn", full_html=True, default_width="100%", div_id="figure",
                         config={"responsive": True, "displaylogo": False})
    select = (f'<div style="margin: 0 0 6px 0; font-family: {FAMILY}; font-size: 12px;">'
              f'<label for="view" id="view-label">{label} </label><select id="view" style="{SELECT_STYLE}">{options}</select>'
              '<span id="view-status" style="margin-left: 8px; color: #555;"></span></div>')
    script = f"<script>const VIEWS = {views};{DROPDOWN_JS}</script>"
    html = html.replace("<body>", "<body>" + select, 1).replace("</body>", script + "</body>", 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
