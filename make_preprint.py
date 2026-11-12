"""
Turn a merged copy of the TMLR site template into the de-anonymized preprint site.

    python -I make_preprint.py SITE_SRC

SITE_SRC is a copy of tmlr-beyond-pdf-author-kit/tmlr_do_not_modify/ after compile_submission.py's
merge step, so _under_review/submission.md is already there with its math converted. This rewrites
that copy in place:

  - the paper becomes preprint.md, served at the site root (the workflow puts it at /preprint/),
    with the authors, affiliations, repository URL, and end matter of the npj preprint
  - the navbar reads Precision Neurotherapeutics Lab instead of TMLR Beyond PDF (_preprint.yml),
    and the tab shows the lab favicon (preprint_assets/lab-favicon.png, from precisionneuro.stanford.edu)
  - the citation box cites a preprint, and the byline lists every affiliation of an author
  - the byline drops Distill's "Submitted: Not submitted yet." column, and the authors and
    affiliations take its width

submission.md stays anonymous and is the only copy of the paper text: edit it, never the output.
Every replacement must match exactly once, so a reworded source fails the build instead of
silently shipping an anonymous or half-anonymous preprint.
"""
import os
import shutil
import sys

LAB = 'Precision Neurotherapeutics Lab'
REPO = 'https://github.com/PrecisionNeuroLab/Symptom-Vectors-for-Depression'
FAVICON = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'preprint_assets', 'lab-favicon.png')

# Authors and affiliations as in the npj manuscript (nature_preprint.tex).
STANFORD_PSY = 'Department of Psychiatry & Behavioral Sciences, Stanford University School of Medicine'
WU_TSAI = 'Wu Tsai Neurosciences Institute, Stanford University'
MIRECC = ('Sierra Pacific Mental Illness, Research, Education, and Clinical Center (MIRECC), '
          'Veterans Affairs Palo Alto Healthcare System')
INDEPENDENT = 'Independent Researcher, Seattle, WA'

# Page CSS (the distill layout's _styles). On wide screens the byline is four columns: authors and
# affiliations span two, "Submitted" one, and one is empty. Drop "Submitted"; give affiliations 3/4.
BYLINE_CSS = '''\
d-byline .byline > div:not(.authors-affiliations) { display: none; }
@media (min-width: 768px) {
  d-byline .byline .authors-affiliations { grid-column: 1 / -1; grid-template-columns: 1fr 3fr; }
}
'''
AUTHORS = [
    ('Fangyi Zhu', [STANFORD_PSY, WU_TSAI]),
    ('Ajay Subramanian', [STANFORD_PSY, WU_TSAI]),
    ('Allison Constant', [STANFORD_PSY]),
    ('Camille Wang', [STANFORD_PSY, WU_TSAI]),
    ('Ravish Gupta', [INDEPENDENT]),
    ('Corey J. Keller', [STANFORD_PSY, WU_TSAI, MIRECC]),
]

AUTHOR_NOTE = ('<p><i>Fangyi Zhu and Ajay Subramanian contributed equally to this work. '
               'Correspondence: Fangyi Zhu, '
               '<a href="mailto:contact@fangyizhu.com">contact@fangyizhu.com</a>.</i></p>\n\n')

# Declarations from the npj manuscript, withheld from the anonymous submission.
END_MATTER = """\
## Acknowledgements

We thank P. J. Hansel for personally providing the Apple Silicon hardware on which the experiments were run and for valuable feedback on the manuscript, Connor Stone (University of Toronto) for collaboration on the `pted` energy-distance package, and Ethan Andrew Solomon (Stanford University) for valuable feedback on the manuscript.

## Funding

Not applicable.

## Competing Interests

F.Z. and R.G. hold stock in Alphabet Inc., the parent company of Google, which developed the Gemma model evaluated in this study. R.G. is employed by Commerce.com, Inc.; the research reported here was conducted independently, unrelated to and unsupported by the author's employment, and the views expressed are the author's own. A.S. and C.J.K. hold stock in Orchard Neuro. C.J.K. is a consultant for Salma Health, Flow Neuroscience, Kyron Medical, and Constellation Systems. A.C. and C.W. declare no competing interests.

## Author Contributions

All authors contributed to writing the manuscript. F.Z. wrote the analysis code, preprocessed all three datasets, conducted all experiments, and wrote the manuscript. A.S. conceived the study and participated in all experiment designs. A.C. and C.W. identified symptom categories using clinical expertise and reviewed and edited the manuscript. R.G. helped with fact checking, reviewing, and editing parts of the manuscript. C.J.K. designed part of the experiments, data analysis and presentation, and supervised the study.

"""
END_MATTER_TOC = ''.join(f'  - name: {name}\n' for name in
                         ['Acknowledgements', 'Funding', 'Competing Interests', 'Author Contributions'])


def authors_yaml():
    out = ['authors:']
    for name, affils in AUTHORS:
        out.append(f'  - name: {name}')
        out.append('    affiliations:')
        out += [f'      - name: "{a}"' for a in affils]
    return '\n'.join(out) + '\n'


def styles_yaml():
    return '_styles: |\n' + ''.join(f'  {line}\n' for line in BYLINE_CSS.splitlines())


def replace_once(text, old, new, where):
    n = text.count(old)
    if n != 1:
        sys.exit(f'make_preprint: expected exactly one {old[:60]!r} in {where}, found {n}')
    return text.replace(old, new)


def paper(src):
    p = replace_once(src, '# Anonymized for review\nauthors:\n  - name: Anonymous\n    affiliations:\n'
                          '      name: Anonymous\n', authors_yaml() + 'permalink: /index.html\n' + styles_yaml(), 'front matter')
    p = replace_once(p, '  - name: GenAI Usage Disclosure\n',
                     '  - name: GenAI Usage Disclosure\n' + END_MATTER_TOC, 'toc')
    p = replace_once(p, 'is provided in an anonymized repository for review: [ANONYMIZED REPOSITORY URL].',
                     f'is available at [{REPO.removeprefix("https://")}]({REPO}).', 'Reproducibility Statement')
    p = replace_once(p, '\n## Abstract\n', '\n' + AUTHOR_NOTE + '## Abstract\n', 'body')
    p = replace_once(p, '\n## Appendix A ', '\n' + END_MATTER + '## Appendix A ', 'body')
    for leftover in ['Anonymous', 'nonymized repository', 'ANONYMIZED']:
        if leftover in p:
            sys.exit(f'make_preprint: {leftover!r} still in the preprint')
    return p


def layout(src):
    # Cite a preprint, not TMLR Beyond PDF.
    s = replace_once(src, '@inproceedings{', '@misc{', 'distill.html')
    s = replace_once(s, 'booktitle = {TMLR Beyond PDF},', 'note = {Preprint},', 'distill.html')
    s = replace_once(s, '"${title}", TMLR Beyond PDF, 2026.', '"${title}", Preprint, 2026.', 'distill.html')
    # The template takes one affiliation per author; list them all.
    return replace_once(s, """\
          "affiliations": [
            {
              "name": "{{ author.affiliations.name }}",
              "url": "{{ author.affiliations.url }}"
            }
          ]""", """\
          "affiliations": [
            {% for affil in author.affiliations -%}
            {
              "name": "{{ affil.name }}",
              "url": "{{ affil.url }}"
            }{% if forloop.last == false %},{% endif %}
            {% endfor %}
          ]""", 'distill.html')


def rewrite(path, fn):
    with open(path, encoding='utf-8') as f:
        text = fn(f.read())
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)


def main(site):
    with open(os.path.join(site, '_under_review', 'submission.md'), encoding='utf-8') as f:
        text = paper(f.read())
    with open(os.path.join(site, 'preprint.md'), 'w', encoding='utf-8') as f:
        f.write(text)
    # Nothing anonymous, and no redirect to it, in the preprint site.
    shutil.rmtree(os.path.join(site, '_under_review'))
    os.remove(os.path.join(site, 'index.md'))
    rewrite(os.path.join(site, '_layouts', 'distill.html'), layout)
    shutil.copyfile(FAVICON, os.path.join(site, 'assets', 'img', 'lab-favicon.png'))
    with open(os.path.join(site, '_preprint.yml'), 'w', encoding='utf-8') as f:
        f.write(f'title: {LAB}\n'
                'icon: lab-favicon.png\n'
                'description: Interpretable Symptom Vectors for Depression in Large Language Models\n')
    print(f'make_preprint: wrote {site}/preprint.md')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
