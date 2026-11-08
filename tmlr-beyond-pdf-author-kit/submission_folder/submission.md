---
layout: distill
title: Interpretable Symptom Vectors for Depression in Large Language Models
htmlwidgets: true

# Anonymized for review
authors:
  - name: Anonymous
    affiliations:
      name: Anonymous

bibliography: submission.bib

toc:
  - name: Abstract
  - name: Introduction
  - name: Related Work
  - name: Data and Models
    subsections:
      - name: Text corpora
      - name: Text anonymization and standardization
      - name: Residual stream extraction
  - name: Are symptom groups separable in residual streams?
  - name: Does projection onto Symptom Vectors yield clinician-agreeing symptom scores?
  - name: Does a single Depression Vector distinguish depressive from non-depressive text?
  - name: Summary across models
    subsections:
      - name: Pre-trained versus instruction-tuned models
  - name: Discussion and Conclusion
  - name: Limitations and Future Work
  - name: Broader Impact Statement
  - name: Reproducibility Statement
  - name: GenAI Usage Disclosure
  - name: Appendix A Core Clinical symptom descriptions
  - name: Appendix B PERMANOVA and PERMDISP statistics at the gated peaks
  - name: Appendix C Within-source clinician agreement
---

## Abstract

Patients with depression present with diverse symptom profiles, yet research and clinical practice routinely reduce this variation to a single severity score. Large language models (LLMs) could capture individual symptoms from patient narratives. However, how depressive symptoms are represented inside LLMs remains poorly understood, limiting clinical trust. To examine whether internal model activations match clinician judgment, we analyzed the residual streams of ten open-weight models from three families (Gemma-3, Mistral-Small, OLMo-2) using interpretability techniques. When activations for clinical symptom descriptions were grouped by symptom category, the groups occupied distinct regions of activation space in every model. In eight of the ten models, the symptom groups were most separable at 20–34% of network depth, at a residual stream layer which we call the operating layer. At each model's operating layer, we built Symptom Vectors from the clinical descriptions and projected unseen naturalistic narratives onto these vectors. The resulting per-symptom coefficients matched clinician-assigned categories across mood, somatic, and suicidality axes in all ten models. Furthermore, a single Depression Vector distinguished held-out depressive from non-depressive text in every model (AUC = 0.730–0.965), capturing a population-level valence direction that could serve as a preliminary gate before symptom projection. These results reveal a clinician-agreeing symptom signal in internal activations across model families. With clinical validation, this pipeline could support depression symptom-profiling tools built on model activations.

## Introduction

Major depressive disorder spans highly diverse symptom profiles, biological underpinnings, and treatment trajectories. This multidimensional complexity remains one of the most significant barriers to effective psychiatric care. In the STAR*D cohort, 3,703 patients diagnosed with major depressive disorder reported 1,030 unique symptom profiles <d-cite key="fried2015depression"></d-cite>. Standard questionnaires like PHQ and HAM-D prioritize brevity; they collapse rich symptom dimensions into a single score while overlooking symptoms that matter to patients <d-cite key="chevance2020identifying"></d-cite>. These discarded nuances, however, are preserved when patients describe their own experiences in naturalistic language. Symptom extraction from such language could help differentiate patients with divergent underlying profiles and link linguistic markers to the neural and physiological biomarkers central to psychiatric neuroscience. It has long been a goal of clinical NLP.

Recent LLMs, which process text in context, have renewed interest in this goal. Existing systems, however, read their assessments from generated outputs. These outputs yield broad labels rather than symptom-level signals and are an unreliable window into model internals (see Related Work). Interpretability techniques offer a route to per-symptom signals read directly from the model's internal representations. To our knowledge, no prior work has mapped a clinically validated depression symptom taxonomy onto per-symptom axes in an LLM's internal activations.

As an LLM processes text, it builds internal representations in a high-dimensional activation space known as the residual stream. Prior work suggests emotions are encoded as linear directions in this space; depression symptom categories may be similarly represented. Applying interpretability techniques to ten pre-trained and instruction-tuned models from three families, we show that these categories (mood/emotional/cognitive, somatic, and suicidality) occupy distinct regions of activation space in every model, most consistently at 20–34% of network depth in eight of the ten. In every model, projecting held-out naturalistic narratives onto Symptom Vectors built from clinical assessment instruments yields coefficients that predict clinician category annotations, and a single Depression Vector distinguishes depressive from non-depressive text. This study establishes a pipeline for extracting interpretable symptom signals directly from an LLM's internal state, preserving the nuanced profiles that brief severity scores obscure. With longitudinal validation, such granular signals could support symptom tracking for research and, eventually, clinical use.

## Related Work

**LLMs for depression assessment.** Existing LLM-based systems for mental health typically produce broad diagnostic labels or sum scores rather than symptom-level assessments <d-cite key="yang2024mentallama,xu2024mentalllm,lamichhane2023evaluation"></d-cite>, or require task-specific fine-tuning for symptom-level analysis <d-cite key="weber2025finetuned"></d-cite>. Generated outputs are also an unreliable window into model internals: GPT-4's internal schema of depression underemphasizes suicidality and overemphasizes psychomotor symptoms relative to expert and self-report ratings <d-cite key="ganesan2026explaining"></d-cite>. We instead read symptom signals from the internal activations of unmodified open-weight models.

**Linear representations of concepts and affect.** Semantic categories are encoded as approximately linear directions in LLM residual streams <d-cite key="park2025geometry"></d-cite>. Emotion representations follow the same pattern, with sentiment localized to specific layers in LLaMA <d-cite key="hollinsworth2024sentiment,dipalma2025llamas"></d-cite>, and the emotional information models pick up from text can be traced to specific, interpretable internal signals <d-cite key="tak2025mechanistic"></d-cite>. This work characterizes coarse valence. Here, we ask whether a finer, clinically derived symptom taxonomy is represented in the same way.

**Semantic projection.** Grand et al. <d-cite key="grand2022semantic"></d-cite> showed that human feature judgments can be reproduced by projecting word embeddings onto axes defined by antonym pairs. We adapt this framework to class-centroid axes in a transformer residual stream, projected jointly with a Gram-inverse correction (described below).

## Data and Models

As shown in Figure 1, our analysis consists of four stages: (1) extracting residual stream activations for symptom descriptions, (2) selecting the operating layer where the three symptom groups are most separable across distance metrics, (3) constructing Symptom Vectors at that layer to project held-out text, and (4) contrasting the Depression Centroid and the Positive Affect Centroid to derive a single Depression Vector. Every stage is run identically on each of the ten models.

<figure id="fig-pipeline" style="text-align: center; margin: 20px 0;">
  <img src="{{ 'assets/img/submission/1.pipeline.png' | relative_url }}" style="max-width: 100%; height: auto;" alt="Analysis pipeline diagram in four panels">
  <figcaption style="text-align: left; margin-top: 12px;"><b>Figure 1: Analysis pipeline</b>, applied to each model. <b>(a)</b> Residual stream activations are extracted at every layer for symptom descriptions from three groups. <b>(b)</b> Distance matrices are tested with PERMANOVA and PERMDISP at each layer to select the operating layer ℓ*. <b>(c)</b> Symptom Vectors are built from the group centroids at ℓ*, and naturalistic text is projected onto them to obtain coefficients β. <b>(d)</b> The Depression Vector is the difference between the Depression Centroid and the Positive Affect Centroid. Each model has <i>n</i> transformer blocks (34–64 across the ten models).</figcaption>
</figure>

### Text corpora

The *Core Clinical* corpus was used to construct the Symptom Vectors. It comprised symptom descriptions extracted from six validated depression assessment instruments: the DSM-5 <d-cite key="apa2013dsm5"></d-cite>, ICD-10 <d-cite key="who1992icd10"></d-cite>, Hamilton Depression Rating Scale (HAM-D) <d-cite key="hamilton1960rating"></d-cite>, Montgomery-Åsberg Depression Rating Scale (MADRS) <d-cite key="montgomery1979new"></d-cite>, Patient Health Questionnaire (PHQ-9) <d-cite key="kroenke2001phq9"></d-cite>, and the patient-derived PROCEED outcome set <d-cite key="chevance2020identifying"></d-cite>. Each instrument was segmented into shorter, self-contained excerpts. Each excerpt was assigned to a single symptom group by one psychiatrist author; a second psychiatrist independently reviewed all assignments against DSM-5 item definitions. Disagreements were resolved through joint review until the two annotators reached consensus, and a final label was assigned only by agreement. This yielded 51 excerpts (19 mood, 24 somatic, 8 suicidality) distributed across DSM-5 (3), ICD-10 (3), HAM-D (16), MADRS (12), PHQ-9 (9), and PROCEED (8).

We also assembled a *Naturalistic* corpus as held-out texts for Symptom Vector evaluation. This corpus comprised first-person and clinical-narrative descriptions of the same three symptom groups, drawn from three sources: William Styron's memoir *Darkness Visible: A Memoir of Madness* <d-cite key="styron1990darkness"></d-cite>, the *Handbook of Depression*, 3rd Edition <d-cite key="gotlib2014handbook"></d-cite>, and ReDSM5, a corpus of Reddit posts annotated at the sentence level against the nine DSM-5 depression symptoms by a licensed psychologist <d-cite key="bao2025redsm5"></d-cite>. We selected *Darkness Visible* for its sustained first-person account of lived depressive experience and the *Handbook of Depression* for its third-person clinical description of symptoms; ReDSM5 was used to supplement the books. *Darkness Visible* and *Handbook* excerpts were annotated using the same two-author categorize-then-review procedure as for the Core Clinical corpus. ReDSM5 posts were mapped onto the three symptom groups from their DSM-5 labels. To balance the symptom groups, we retained every book-sourced passage and added randomly selected ReDSM5 posts to each group until its count matched that of the largest book-sourced group. Table 1 reports the resulting sample counts.

<p style="text-align: center; margin-bottom: 0.3em;"><b>Table 1:</b> Naturalistic corpus composition. Excerpt counts by source and symptom group.</p>

| Source | Mood | Somatic | Suicidality | Source total |
|:--|--:|--:|--:|--:|
| Darkness Visible | 33 | 11 | 13 | 57 |
| Handbook of Depression | 108 | 42 | 22 | 172 |
| ReDSM5 | — | 88 | 106 | 194 |
| **Symptom total** | **141** | **141** | **141** | **423** |

Excerpts of the Core Clinical and Naturalistic corpora were annotated with one of three symptom groups: mood/emotional/cognitive (hereafter "mood"; DSM-5 A1, A2, A7, A8: depressed mood, anhedonia, guilt or shame, concentration difficulties); somatic (DSM-5 A3–A6: appetite or weight change, sleep disturbance, psychomotor change, and fatigue in the context of depression); and suicidality (DSM-5 A9: active suicidal ideation with or without specific plan, and passive death wishes). The mood and somatic groups were chosen to reflect depression's distinct impact on the mind or body, following prior meta-analytic work on clinical instruments <d-cite key="shafer2006metaanalysis"></d-cite>. Given its clinical significance and risk profile, suicidality was assigned its own category.

The *Positive Affect* corpus comprises nine short excerpts expressing joy, love, wonder, aspiration, and optimism, opposite in valence to the depression corpora. Its sources span genres so that its centroid reflects positive valence rather than any one style: translated classical Chinese poetry (Li Bai, Cao Cao), literary fiction (Tolstoy, García Márquez, Rothfuss), popular science (Dawkins), political oratory (Kennedy), and civic prose (the U.S. Constitution).

To contrast depressive text with affectively positive language, we included *HappyDB*, a corpus of crowdsourced descriptions of happy moments <d-cite key="asai2018happydb"></d-cite>. We took the first 423 non-duplicate entries to match the size of the Naturalistic corpus.

### Text anonymization and standardization

A pilot experiment on the unedited Core Clinical corpus placed PROCEED's residual stream activations far away from those of every other clinical instrument, regardless of clinical similarity. This distance came from PROCEED's bullet-point format. To prevent stylistic confounds such as format, pronouns, and register from being picked up as signal, we rewrote every excerpt into a standard third-person register using Anthropic Claude Opus 4 with one prompt: "Rewrite the following text into natural language as a description of a third person. Correct any grammar mistakes." The edits changed format only: standardizing pronouns, removing severity ratings from questionnaires, correcting grammar in ReDSM5 and HappyDB, restructuring PROCEED fragments into prose, and removing names and identifying details. Clinical content was left unchanged, and the authors reviewed and corrected every rewrite.

The full set of rewritten excerpts in the Core Clinical corpus is provided in Appendix A. Table 2 shows representative examples of the rewriting from each corpus except ReDSM5, whose license prohibits reproducing posts.

<p style="text-align: center; margin-bottom: 0.3em;"><b>Table 2:</b> Text standardization examples. Original and standardized excerpts from each corpus except ReDSM5.</p>

| Dataset | Original | Standardized |
|:--|:--|:--|
| Core Clinical | INSOMNIA – Delayed (Waking in early hours of the morning and unable to fall asleep again)<br>0 = Absent<br>1 = Occasional<br>2 = Frequent | Delayed Insomnia: The person wakes up in the early hours of the morning and is unable to fall asleep again. |
| Naturalistic (Darkness Visible) | I now see, as a means to calm the anxiety and incipient dread that I had hidden away for so long somewhere in the dungeons of my spirit. | They now see this as a means to calm the anxiety and incipient dread that they had hidden away for so long somewhere in the dungeons of their spirit. |
| Naturalistic (Handbook of Depression) | Individuals with depression exhibit cognitive deficits and biased processing of emotional material. | They exhibit cognitive deficits and biased processing of emotional material. |
| Positive Affect | He spoke gently, laughed often, and never exercised his wit at the expense of others. | They speak gently, laugh often, and never exercise their wit at the expense of others. |
| HappyDB | I went on a successful date with someone I felt sympathy and connection with. | They went on a successful date with someone they felt sympathy and a connection with. |

### Residual stream extraction

We analyzed ten open-weight models, each the official Hugging Face release: Gemma-3 at 4B, 12B, and 27B parameters, each pre-trained (PT) and instruction-tuned (IT) <d-cite key="gemmateam2025gemma3"></d-cite> (34, 48, and 62 transformer blocks; residual stream widths 2,560, 3,840, and 5,376); Mistral-Small-24B Base and Instruct (2501 release; 40 blocks, width 5,120); and OLMo-2-32B base and Instruct (0325 release; 64 blocks, width 5,120) <d-cite key="olmo2024olmo2"></d-cite>. Gemma-3's scales vary within its family, and the five PT/IT pairs test instruction tuning. Instruction-tuned models received the same raw text as the PT models, without a chat template.

Each excerpt was tokenized with the model's own tokenizer and processed in a single forward pass with Hugging Face Transformers <d-cite key="wolf2020transformers"></d-cite> on PyTorch <d-cite key="paszke2019pytorch"></d-cite>. Forward hooks recorded the input embeddings and the residual stream activation after every transformer block. The initial `<bos>` activation was discarded at every hook point to mitigate attention-sink and massive-activation effects <d-cite key="xiao2024attentionsinks,sun2024massive"></d-cite>; for OLMo-2, whose tokenizer adds no beginning-of-sequence token, we prepended its `bos_token` so the same position could be discarded.

<div id="model-picker" style="display: none; position: fixed; right: 16px; top: 16px; z-index: 1000; background: white; border: 1px solid #ccc; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.15); padding: 6px 10px; font: 13px Arial, sans-serif; color: black;"><label>Model <select id="model-select" style="font-size: 1em; padding: 2px 4px; max-width: 60vw;"></select></label></div>

<style>
/* Print / PDF: hide the floating selector and show every model's Table B1, since the archival PDF cannot switch models. */
@media print {
  #model-picker { display: none !important; }
  .model-views [data-model][hidden] { display: block !important; break-inside: avoid; margin-top: 1em; }
}
</style>

<script>
// One floating model selector for every figure and table on the page. Each figure iframe
// (figure_style.write_tmlr_dropdown_html) posts {type: 'tmlr-model-ready', labels} on load: the
// first fills the selector, each is told the current model, and the selector shows (top right,
// under the TMLR navbar, beside the figure column when there is room) while any of them is on
// screen. Changing it sets every figure.
(function () {
  var picker = document.getElementById('model-picker'), select = document.getElementById('model-select');
  document.body.appendChild(picker);  // fixed to the viewport, clear of any transformed ancestor
  // Inline per-model tables (.model-views) follow the selector too: one [data-model] child shows.
  // They sit later in the page than this script, so they are looked up on each place() call.
  var frames = [];
  function showInline(model) {
    document.querySelectorAll('.model-views [data-model]').forEach(function (v) {
      v.hidden = v.getAttribute('data-model') !== model;
    });
  }
  function place() {
    var nav = document.getElementById('navbar');  // fixed-top or sticky-top in the TMLR layout
    picker.style.top = (nav ? Math.max(0, nav.getBoundingClientRect().bottom) : 0) + 12 + 'px';
    var targets = frames.concat(Array.prototype.slice.call(document.querySelectorAll('.model-views')));
    var onScreen = targets.filter(function (f) {
      var r = f.getBoundingClientRect();
      return r.bottom > 0 && r.top < window.innerHeight;
    });
    picker.style.display = onScreen.length && select.options.length ? 'block' : 'none';
    if (!onScreen.length) return;
    // Beside the figure column when the margin has room for it, else at the window edge (phones).
    var edge = Math.max.apply(null, onScreen.map(function (f) { return f.getBoundingClientRect().right; }));
    var fits = edge + 16 + picker.offsetWidth <= window.innerWidth - 16;
    picker.style.left = fits ? edge + 16 + 'px' : 'auto';
    picker.style.right = fits ? 'auto' : '16px';
  }
  window.addEventListener('scroll', place, {passive: true});
  window.addEventListener('resize', place);
  function tell(win) { if (win && select.value) win.postMessage({type: 'tmlr-model', model: select.value}, '*'); }
  window.addEventListener('message', function (e) {
    if (!e.data || e.data.type !== 'tmlr-model-ready') return;
    if (!select.options.length) e.data.labels.forEach(function (l) { select.add(new Option(l, l.split(' · ')[0])); });
    document.querySelectorAll('iframe').forEach(function (f) {
      if (f.contentWindow === e.source && frames.indexOf(f) < 0) frames.push(f);
    });
    place();
    tell(e.source);
  });
  select.addEventListener('change', function () {
    showInline(select.value);
    document.querySelectorAll('iframe').forEach(function (f) { tell(f.contentWindow); });
  });
})();
</script>

<script>
// Size each figure iframe to its content: no inner scrollbar and no slack under the figure,
// whatever the fonts or zoom. The height attributes stay as the size before the figure loads.
(function () {
  function fit(f) {
    try {
      var body = f.contentDocument && f.contentDocument.body;
      if (!body || !body.offsetHeight) return;
      var cs = f.contentWindow.getComputedStyle(body);
      f.style.height = Math.ceil(body.offsetHeight + parseFloat(cs.marginTop) + parseFloat(cs.marginBottom)) + 'px';
    } catch (e) {}  // a cross-origin figure keeps its height attribute
  }
  function watch(f) {
    fit(f);
    try { new ResizeObserver(function () { fit(f); }).observe(f.contentDocument.body); } catch (e) {}
  }
  function init() {
    document.querySelectorAll('iframe').forEach(function (f) {
      f.addEventListener('load', function () { watch(f); });
      watch(f);
    });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
</script>

## Are symptom groups separable in residual streams?

### Method

Different residual stream layers encode different information. To identify where each model distinguishes depression symptoms, we quantified symptom-group separability at each residual stream layer on the Core Clinical corpus using PERMANOVA. We then used the model's operating layer (defined below) for the Symptom Vector extraction.

#### Pairwise distance matrices

For a given input, the residual stream activation is a variable-length sequence of high-dimensional vectors (2,560–5,376 dimensions across the ten models), where each token’s vector depends on all preceding tokens through self-attention. Tokens within an activation are therefore not independent samples; each activation has to be compared as a whole. There is no established method for measuring the distance between residual streams generated by prompts of different lengths for the purpose of comparing textual meaning. We tested the efficacy of eight distance-metric × normalization combinations: Centroid Cosine Distance, Centroid Euclidean Distance, Earth Mover's Distance, and Energy Distance, each computed on both raw and unit-normalized residual stream activations, at every residual stream layer $$\ell$$ of each model.

These metrics are not directly comparable, so we reduced each to a common representation: a pairwise distance matrix. For a given metric and layer, entry $$(i, j)$$ is the distance between the residual stream activations of prompts $$i$$ and $$j$$ from the Core Clinical corpus. Classical ANOVA and MANOVA need fixed-length coordinate vectors and normally distributed underlying data; residual streams satisfy neither requirement. PERMANOVA operates directly on the pairwise distances, and its pseudo-$$F$$ statistic is sensitive to the distance between group centroids. Because pseudo-$$F$$ is a ratio of between- to within-group variation, it is unaffected by the absolute scale of the input distances. PERMANOVA can thus be used to compare different metrics directly.

The centroid metrics use the mean of all token activations. Let $$h_{x,t}$$ denote the residual stream activation of token $$t$$ of passage $$x$$ at layer $$\ell$$, for a passage of $$T_x$$ tokens. Excluding `<bos>`, the passage centroid is

$$
\bar{h}_x = \frac{1}{T_x} \sum_{t=1}^{T_x} h_{x,t}.
$$

Distribution-based metrics (Earth Mover's, Energy) were computed directly between full token distributions, using SciPy <d-cite key="virtanen2020scipy"></d-cite> and the `pted` energy-statistic package <d-cite key="stone2026pted,szekely2013energy"></d-cite>, respectively.

#### PERMANOVA with the PERMDISP eligibility gate

Using the pairwise distance matrices, we assessed the separability of residual stream activations across symptom groups with PERMANOVA, gated by PERMDISP. At each layer and for each metric, we computed a PERMANOVA pseudo-$$F$$ statistic <d-cite key="anderson2001permanova"></d-cite>:

$$
\text{pseudo-}F = \frac{\sum(\text{Between-Group Distance})^2 / (g-1)}{\sum(\text{Within-Group Distance})^2 / (n-g)},
$$

where $$g = 3$$ is the number of symptom groups and $$n$$ is the total number of excerpts.

PERMANOVA is sensitive not only to differences in group centroid location but also to differences in within-group multivariate dispersion: a group whose excerpts are widely spread can produce a significant pseudo-$$F$$ even when its centroid coincides with the others. To screen out layers where this happens, we computed the Permutational Analysis of Multivariate Dispersions (PERMDISP) <d-cite key="anderson2006permdisp"></d-cite>, which tests the null hypothesis of equal within-group dispersion, at every layer, and define the *eligible region* of a metric as the set of layers at which PERMDISP is non-significant ($$p > 0.05$$). All distance statistics were computed with the scikit-bio package <d-cite key="aton2026scikitbio"></d-cite> (PERMDISP with `test="centroid"`), each on 9,999 permutations with a fixed seed.

#### Operating layer selection

For each metric, the *gated peak* (the operating peak in Figure 2) is the layer of highest PERMANOVA pseudo-$$F$$ within the eligible region. Because models differ in depth, each layer is also reported as relative depth (layer divided by the number of transformer blocks).

A model's *operating layer* $$\ell^\ast$$ is the layer at which the largest number of metrics reach their gated peak, with ties broken in favor of the layer at which more metrics pass the PERMDISP gate. At $$\ell^\ast$$, the *max gated pseudo-F* is the highest pseudo-$$F$$ among all metrics passing the PERMDISP gate at that layer. Each model's Symptom Vectors, symptom projection, and Depression Vector are built at its own operating layer.

### Results

At every layer of each model, we tested whether the three symptom groups occupy distinct regions of activation space. Every result below is reported for all ten models. The model-specific figures (Figures 2–4) show one model at a time; the floating Model selector at the upper right switches all of them to another model at once. Table 3 summarizes every model across experiments.

For each of the eight distance-metric × normalization combinations, we located its gated peak, the layer of strongest symptom-group separation that passes the PERMDISP gate (Figure 2; Appendix B). In all ten models, every metric had an eligible layer, and every gated peak was significant (PERMANOVA $$p \le 0.0313$$).

<figure id="fig-separation" style="text-align: center; margin: 20px 0;">
  <iframe src="{{ 'assets/html/submission/82.multimodel_separation.html' | relative_url }}" width="100%" height="808" style="border: none;" title="Per-layer PERMANOVA pseudo-F for eight distance metrics"></iframe>
  <figcaption style="text-align: left;"><b>Figure 2: Symptom separability across layers, with each metric's gated peak.</b> PERMANOVA pseudo-<i>F</i> at every layer for the eight distance metrics. Shaded bands mark layers excluded by the PERMDISP gate. Green marks each metric's highest eligible peak; gray marks a higher but excluded peak, where one exists.</figcaption>
</figure>

The gated peaks tended to appear at an early operating layer in most models: at 20–34% of depth in eight of the ten, and at 59% in the Gemma-3-4B pair. Between two and five of the eight metrics reached their gated peak at the operating layer, and the max gated pseudo-$$F$$ ranged from 5.41 to 8.45 (Table 3).

## Does projection onto Symptom Vectors yield clinician-agreeing symptom scores?

### Method

We adapt the semantic projection framework of Grand et al. <d-cite key="grand2022semantic"></d-cite>, in which the line from "small" to "big" in word-embedding space tracks perceived size, with three changes. The axes are class centroids rather than antonym pairs, because symptom categories have no natural antonyms. The three axes are projected jointly, with a Gram-inverse correction that isolates each axis's own contribution. And the projected representations are residual stream activations at the operating layer rather than static word vectors.

**Symptom Vectors.** At the operating layer $$\ell^\ast$$, we built one Symptom Vector $$v_s$$ for each symptom group $$s$$ (mood, somatic, suicidality): the centroid of all residual stream activations from that group's Core Clinical excerpts, excluding `<bos>`. Stacking the three gives the basis

$$
V = [\, v_{\text{mood}};\ v_{\text{somatic}};\ v_{\text{suicidality}} \,] \in \mathbb{R}^{3 \times d},
$$

where $$d$$ is the model's residual stream width. We call the span of $$V$$ the *symptom subspace*.

**Projection.** Depression symptoms co-occur in text, so the Symptom Vectors overlap, and a plain projection would count their shared signal on every axis. We correct for this with the Gram matrix $$G = V V^\top$$. For a passage $$x$$ with centroid $$\bar{h}_x$$ at the operating layer, the coefficients are

$$
\beta(x) = G^{-1} V \, \bar{h}_x = (\beta_{\text{mood}}, \beta_{\text{somatic}}, \beta_{\text{suicidality}}).
$$

When the basis is orthonormal, $$G^{-1}$$ is the identity and this reduces to Grand et al.'s independent projection onto each axis. A positive $$\beta_s$$ means the passage points along $$v_s$$, and a negative one means it points against it. Because $$V$$ is fixed, coefficients from different passages and corpora share one coordinate system. The condition number $$\kappa(G)$$, the ratio of $$G$$'s largest to smallest eigenvalue, measures how collinear the Symptom Vectors are; we report it for each model in the Results.

**Evaluation.** We projected every Core Clinical and Naturalistic passage. The Core Clinical projections are an in-sample consistency check, and the held-out Naturalistic projections test generalization. Group differences on each axis were tested with Dunn's post-hoc test, Benjamini–Hochberg corrected within each corpus.

### Results

In each model, we projected every Core Clinical and Naturalistic passage onto the three Symptom Vectors at the operating layer and asked whether the coefficients agree with clinician annotation.

In Mistral and OLMo, $$\kappa(G)$$ is about 35 (33.5–37.1), so the symptom axes overlap substantially but remain clearly distinguishable. In every Gemma-3 model, $$\kappa(G)$$ is 8,959–18,144, meaning the three Symptom Vectors are nearly parallel and differ only along small residual directions (Table 3). This pattern is consistent across model sizes and between pre-trained and instruction-tuned variants, which suggests it reflects how each model family organizes its residual stream rather than scale or instruction tuning.

Even in models where the axes are nearly parallel, the coefficients agree with clinician annotation. Figure 3 shows the coefficient distributions for each clinician-annotation group within each dataset, with their medians in panel (d). On the in-sample Core Clinical corpus, the excerpts carrying the matching annotation attained the highest median coefficient on each of the three axes in all ten models (Figure 3(a)–(c)).

The same rank ordering held on the held-out Naturalistic corpus: in all ten models, the matching annotation group again ranked first on every axis (Figure 3(d)). In both corpora and every model, each matching group exceeded each of the other two groups on its own axis (all corrected $$p < 0.05$$). On the held-out corpus, the matching group's median led the next-highest group's by 0.46–0.73 on the mood axis, 0.31–0.43 on the somatic axis, and 0.33–0.54 on the suicidality axis across models. Appendix C repeats this analysis within each Naturalistic source to rule out a source confound.

<figure id="fig-coefficients" style="text-align: center; margin: 20px 0;">
  <iframe src="{{ 'assets/html/submission/82.multimodel_coefficient_abc.html' | relative_url }}" width="100%" height="608" style="border: none;" title="Figure 3(a)–(c): per-axis coefficient distributions"></iframe>
  <iframe src="{{ 'assets/html/submission/82.multimodel_coefficient_rank.html' | relative_url }}" width="100%" height="384" style="border: none;" title="Figure 3(d): rank matrices of median coefficients"></iframe>
  <figcaption style="text-align: left;"><b>Figure 3: Semantic projection yields clinician-agreeing symptom coefficients.</b> <b>(a)–(c)</b> Coefficients on the mood, somatic, and suicidality axes, grouped by clinician annotation, for the in-sample Core Clinical corpus (top) and the held-out Naturalistic corpus (bottom). Brackets mark pairwise differences between groups (***, <i>p</i> &lt; 0.001; **, <i>p</i> &lt; 0.01; *, <i>p</i> &lt; 0.05; ns, not significant). <b>(d)</b> Median coefficient for each group on each axis, ranked within each column; bordered cells mark each group's own axis.</figcaption>
</figure>

Figure 4 shows the joint structure of the three coefficients: in every model, clinical and naturalistic passages sharing an annotation cluster together in the symptom subspace. The point clouds are flattened toward a plane, most strongly in the Gemma-3 models, where a passage's three coefficients sum to nearly 1 (mean 0.99–1.03). Contextual activations occupy a narrow cone of the embedding space <d-cite key="ethayarajh2019contextual"></d-cite>, dominated by a few high-magnitude dimensions <d-cite key="timkey2021allbark"></d-cite>, so each Symptom Vector is largely a shared direction plus a smaller symptom-specific offset. The more collinear the vectors, the more a passage's coefficients are forced to sum to about 1, which confines the passages to a plane.

<figure id="fig-subspace" style="text-align: center; margin: 20px 0;">
  <iframe src="{{ 'assets/html/submission/82.multimodel_3d.html' | relative_url }}" width="100%" height="548" style="border: none;" title="Figure 4: passages in the three-dimensional symptom subspace"></iframe>
  <figcaption style="text-align: left;"><b>Figure 4: Passages sharing a clinician annotation occupy the same region of the symptom subspace.</b> Each point is one passage's three coefficients, colored by clinician annotation; squares are in-sample Core Clinical passages and circles held-out Naturalistic passages.</figcaption>
</figure>

## Does a single Depression Vector distinguish depressive from non-depressive text?

### Method

Because projecting non-depressive text onto Symptom Vectors is not meaningful, we also tested whether the operating layer distinguishes depressive from non-depressive text. We built two centroids of opposite emotional valence:

1. The *Depression* centroid is the centroid of the three Symptom Vectors from the previous experiment, constructed from the Core Clinical corpus. This gives equal weight to all three symptoms:

$$
\mu_{\text{depressive}} = \frac{v_{\text{mood}} + v_{\text{somatic}} + v_{\text{suicidality}}}{3}.
$$

{:start="2"}
2. The *Positive Affect* centroid $$\mu_{\text{positive}}$$ is the token-weighted centroid of the Positive Affect corpus, computed in the same way as $$v_s$$ (excluding `<bos>`).

We then define the *Depression Vector* to be:

$$
v_{\text{depression}} = \mu_{\text{depressive}} - \mu_{\text{positive}}.
$$

The Naturalistic corpus and the HappyDB corpus are reserved as held-out tests that are not included in the construction of these axes.

An excerpt $$x$$ is scored by the cosine similarity between its centroid $$\bar{h}_x$$ and the Depression Vector; we call this the *Depression Score* of excerpt $$x$$:

$$
\text{Score}(x) = \cos(\bar{h}_x, v_{\text{depression}}) = \frac{\bar{h}_x \cdot v_{\text{depression}}}{\lVert \bar{h}_x \rVert \, \lVert v_{\text{depression}} \rVert}.
$$

We compared Depression Scores between the held-out Naturalistic and HappyDB corpora (423 excerpts each) using a one-sided Mann–Whitney $$U$$ test. We report the normalized statistic $$U/(n_1 n_2)$$, where $$n_1$$ and $$n_2$$ are the two corpus sizes. It equals the AUC: the probability that a randomly drawn depressive excerpt outscores a randomly drawn non-depressive one (0.5, no separation; 1.0, complete separation). We computed the same statistic for Core Clinical versus Positive Affect as an in-sample consistency check.

### Results

In each model, we computed the Depression Score of every excerpt to test whether the Depression Vector distinguishes depressive from non-depressive text (Figure 5).

In-sample, Core Clinical excerpts ($$n = 51$$) and Positive Affect excerpts ($$n = 9$$) were completely distinguished by the Depression Score in every model (AUC = 1.000, one-sided Mann–Whitney $$U$$ with normal approximation, $$p = 1.06 \times 10^{-6}$$).

On held-out text, in every model, Naturalistic depressive excerpts ($$n = 423$$) received higher Depression Scores than HappyDB happy-moment descriptions ($$n = 423$$), with AUCs from 0.730 to 0.965 (one-sided Mann–Whitney $$U$$, all $$p < 10^{-30}$$). Eight models reached 0.943–0.965, so that a randomly drawn depressive narrative outscores a randomly drawn happy-moment description about 19 times in 20; the Gemma-3-27B pair was lowest (0.789 PT, 0.730 IT).

The HappyDB entries that score highest are consistent across models: in every model the top-scoring entry describes a bodily or health event, "Their sore throat cleared up" in five models and "They are able to sleep longer than usual" in the other five. Positive in valence, these entries share the somatic vocabulary of the depressive corpora, which the Depression Vector partly carries.

<figure id="fig-depvector" style="text-align: center; margin: 20px 0;">
  <iframe src="{{ 'assets/html/submission/82.multimodel_auc.html' | relative_url }}" width="100%" height="1120" style="border: none;" title="Figure 5: Depression Score by corpus"></iframe>
  <figcaption style="text-align: left;"><b>Figure 5: A single operating-layer Depression Vector distinguishes depressive from non-depressive text.</b> Depression Score of each excerpt by corpus, one panel per model. Core Clinical and Positive Affect were used to build the vector; Naturalistic and HappyDB are held out. Brackets report AUC (***, <i>p</i> &lt; 0.001). Y-axis scales differ by model family.</figcaption>
</figure>

Only the relative position of the boxes within a panel is meaningful, because the scale of the score differs from model to model. In every model, the depressive corpus (blue) scored above its non-depressive pair (pink) within each bracket. The in-sample pair lay furthest apart, and the held-out pair fell between them, with Naturalistic below Core Clinical and HappyDB above Positive Affect; the one exception was Gemma-3-27B-IT, where the Naturalistic median sat marginally above Core Clinical. Because depressive text outscores non-depressive text in all ten models, the layer that separates the three symptom groups also separates depressive from non-depressive text.

## Summary across models

Table 3 collects the per-model results. A nearly parallel symptom basis did not weaken either result: Gemma-3-12B-IT, whose Symptom Vectors have the largest $$\kappa(G)$$ (18,144), also had the highest held-out AUC (0.965). Within Gemma-3, the one family tested at several sizes, scale brought no consistent gain: the 4B models matched the 12B and 27B models on clinician agreement and separation strength, and the held-out AUC was lowest at 27B.

<figure id="tab-models" style="text-align: center; margin: 20px 0;">
  <p style="margin-bottom: 0.3em;"><b>Table 3:</b> Per-model summary at each model's operating layer.</p>
<div style="overflow-x: auto;">
<table style="width: 100%;">
<thead><tr><th style="text-align: left;">Model</th><th style="text-align: center;">Operating layer<br>layer / blocks · depth</th><th style="text-align: center;">Agreeing<br>metrics</th><th style="text-align: center;">Max gated<br>pseudo-<i>F</i></th><th style="text-align: center;">Clinician<br>agreement</th><th style="text-align: center;">Gram<br>κ(<i>G</i>)</th><th style="text-align: center;">Contrastive<br>AUC</th></tr></thead>
<tbody>
<tr><td style="text-align: left;">Gemma-3-27B-PT</td><td style="text-align: center;">21 / 62 · 34%</td><td style="text-align: center;">3 / 8</td><td style="text-align: center;">6.91</td><td style="text-align: center;">3 / 3</td><td style="text-align: center;">14,624.5</td><td style="text-align: center;">0.789</td></tr>
<tr><td style="text-align: left;">Gemma-3-27B-IT</td><td style="text-align: center;">21 / 62 · 34%</td><td style="text-align: center;">5 / 8</td><td style="text-align: center;">8.40</td><td style="text-align: center;">3 / 3</td><td style="text-align: center;">12,484.7</td><td style="text-align: center;">0.730</td></tr>
<tr><td style="text-align: left;">Gemma-3-12B-PT</td><td style="text-align: center;">13 / 48 · 27%</td><td style="text-align: center;">3 / 8</td><td style="text-align: center;">5.91</td><td style="text-align: center;">3 / 3</td><td style="text-align: center;">8,959.2</td><td style="text-align: center;">0.959</td></tr>
<tr><td style="text-align: left;">Gemma-3-12B-IT</td><td style="text-align: center;">14 / 48 · 29%</td><td style="text-align: center;">2 / 8</td><td style="text-align: center;">8.03</td><td style="text-align: center;">3 / 3</td><td style="text-align: center;">18,143.6</td><td style="text-align: center;">0.965</td></tr>
<tr><td style="text-align: left;">Gemma-3-4B-PT</td><td style="text-align: center;">20 / 34 · 59%</td><td style="text-align: center;">3 / 8</td><td style="text-align: center;">6.55</td><td style="text-align: center;">3 / 3</td><td style="text-align: center;">12,491.7</td><td style="text-align: center;">0.945</td></tr>
<tr><td style="text-align: left;">Gemma-3-4B-IT</td><td style="text-align: center;">20 / 34 · 59%</td><td style="text-align: center;">4 / 8</td><td style="text-align: center;">8.45</td><td style="text-align: center;">3 / 3</td><td style="text-align: center;">11,619.5</td><td style="text-align: center;">0.953</td></tr>
<tr><td style="text-align: left;">Mistral-24B-PT</td><td style="text-align: center;">8 / 40 · 20%</td><td style="text-align: center;">5 / 8</td><td style="text-align: center;">5.84</td><td style="text-align: center;">3 / 3</td><td style="text-align: center;">35.5</td><td style="text-align: center;">0.943</td></tr>
<tr><td style="text-align: left;">Mistral-24B-IT</td><td style="text-align: center;">8 / 40 · 20%</td><td style="text-align: center;">5 / 8</td><td style="text-align: center;">6.16</td><td style="text-align: center;">3 / 3</td><td style="text-align: center;">37.1</td><td style="text-align: center;">0.946</td></tr>
<tr><td style="text-align: left;">OLMo-2-32B-PT</td><td style="text-align: center;">16 / 64 · 25%</td><td style="text-align: center;">3 / 8</td><td style="text-align: center;">5.41</td><td style="text-align: center;">3 / 3</td><td style="text-align: center;">33.5</td><td style="text-align: center;">0.945</td></tr>
<tr><td style="text-align: left;">OLMo-2-32B-IT</td><td style="text-align: center;">16 / 64 · 25%</td><td style="text-align: center;">3 / 8</td><td style="text-align: center;">5.46</td><td style="text-align: center;">3 / 3</td><td style="text-align: center;">33.9</td><td style="text-align: center;">0.946</td></tr>
</tbody>
</table>
</div>
</figure>

### Pre-trained versus instruction-tuned models

The five PT/IT pairs isolate the effect of instruction tuning on each of these results.

Instruction tuning left the location of separation largely unchanged: within each pair the operating layer was identical (Gemma-3-27B, Gemma-3-4B, Mistral-Small-24B, OLMo-2-32B) or adjacent (Gemma-3-12B, layers 13 and 14). It did, however, raise the strength of separation: in all five pairs the highest gated pseudo-$$F$$ at the operating layer was higher in the instruction-tuned model (Gemma-3-27B, 8.40 vs. 6.91; Gemma-3-12B, 8.03 vs. 5.91; Gemma-3-4B, 8.45 vs. 6.55; Mistral-Small-24B, 6.16 vs. 5.84; OLMo-2-32B, 5.46 vs. 5.41).

Stronger separation did not translate into better downstream generalization: both the pre-trained and instruction-tuned versions of every model matched the clinician category assignments. Instruction tuning raised the held-out AUC by less than 0.01 in four pairs (Gemma-3-12B, 0.959 PT vs. 0.965 IT; Gemma-3-4B, 0.945 vs. 0.953; Mistral-Small-24B, 0.943 vs. 0.946; OLMo-2-32B, 0.945 vs. 0.946) and lowered it only in Gemma-3-27B (0.789 vs. 0.730).

## Discussion and Conclusion

One challenge in treating depression is its clinical heterogeneity: patients sharing a diagnosis exhibit widely divergent symptom profiles, yet standard questionnaires often collapse this complexity into a single severity score <d-cite key="fried2015depression,chevance2020identifying"></d-cite>. Across ten models from three families, distinct depression symptoms can be read directly from residual streams: the symptom groups separate at an early operating layer (20–34% of depth in eight of the ten models), projection onto Symptom Vectors matches clinician categories on held-out narratives, and a Depression Vector distinguishes depressive from non-depressive text. Because the Symptom Vectors are built only from depressive text, the Depression Vector can serve as a gate that restricts symptom projection to depressive input.

Our findings extend work showing that semantic categories, including sentiment, are encoded as approximately linear directions in LLM residual streams <d-cite key="park2025geometry,hollinsworth2024sentiment,dipalma2025llamas"></d-cite>. While prior work has characterized coarse sentiment, we map a symptom taxonomy drawn from six clinical instruments onto a model's internal representations and extract symptom signals along three axes. Projecting a depressive passage onto the three Symptom Vectors yields a symptom profile: one coefficient each for mood, somatic, and suicidal content.

The black-box nature of large language models is a major barrier to clinical trust and to translating model-based tools into care. We open this box by mapping internal directions onto clinical symptom groups, so a model's representation of a passage can be read symptom by symptom. Extracting symptom coefficients from naturalistic speech also yields a quantitative signal that can be correlated with fMRI, resting EEG, or TMS-evoked potentials <d-cite key="parmigiani2023reliability"></d-cite>. Prior work has mapped continuous behavioral markers to neural activity in naturalistic settings, such as decoding facial expressions for pain <d-cite key="huang2025naturalistic"></d-cite> and mood <d-cite key="kakusa2025distinct"></d-cite>. Symptom coefficients bring this approach to language and can test whether the mood, somatic, and suicidality dimensions extracted by the LLM correspond to dissociable neural signatures. Such signatures would give precision psychiatry symptom-specific treatment targets <d-cite key="cline2026personalizing"></d-cite>. Researchers can map the symptom dimensions of any psychiatric disorder the same way. For AI developers, the pipeline can also flag signs of depression or suicidality in a deployed model's conversations directly from its activations.


## Limitations and Future Work

The Symptom Vectors are built from 51 reference excerpts (8 for suicidality), and the nine DSM-5 symptoms are collapsed into three groups; larger clinical corpora would allow per-symptom vectors and tests of co-occurring symptoms and severity. The operating layer is chosen by a heuristic, the PERMDISP gate has lower power in the small suicidality group, and PERMANOVA $$p$$-values at the gated peaks are not corrected for selection across layers and metrics. The ten models cover three families between 4B and 32B parameters, all tested on English text. We rewrote every excerpt with Claude Opus 4 and limited the edits to format, but the rewriting may still impose a shared style on the clinical and naturalistic corpora, and the pipeline has not been tested on unedited text. The Depression Vector was evaluated by population-level separation only; its use as a thresholded gate is untested. Finally, the projections have not been validated against diagnosis, treatment response, or longitudinal course, which clinical monitoring would require.

## Broader Impact Statement

This study involved the secondary analysis of publicly available, de-identified datasets. Data sources included HappyDB, a corpus of crowdsourced text responses, and ReDSM5, a dataset of public social media posts. Because the research relied exclusively on pre-existing, de-identified, and publicly accessible data, the study was deemed exempt from Institutional Review Board (IRB) review.

Inferring depression symptoms from language carries risks beyond those of the analysis itself. The prospect of an activation-level risk detector raises questions of consent, false-positive burden, and surveillance that any deployment would need to resolve through appropriate governance. The Symptom Vectors and Depression Vector reported here have not been validated against clinical outcomes and should not be used to screen, diagnose, or monitor individuals.

## Reproducibility Statement

All analyses were performed in Python, and code to reproduce every figure and table is provided in an anonymized repository for review: [ANONYMIZED REPOSITORY URL]. All models are public Hugging Face releases, and all permutation tests use 9,999 permutations with a fixed seed. Activations were extracted on Apple Metal Performance Shaders (MPS) and cast to float64 before any distance or centroid computation; centroid distances used PyTorch 2 <d-cite key="ansel2024pytorch2"></d-cite>. The third-person rewritten Core Clinical corpus (excluding copyright-protected source text) and the Positive Affect corpus (public-domain sources) are released with the code, together with the Gram-corrected projection pipeline and, for each of the ten models, the Symptom Vectors, their Gram inverse, and the Depression Vector at the model's operating layer, so that new text can be scored without re-extracting the corpora. The following corpora are withheld: raw clinical instrument text is not redistributed in accordance with source copyright, and the ReDSM5 corpus is not redistributed under its license terms; *Darkness Visible* and the *Handbook of Depression* excerpts are also copyright-protected and not redistributable. Citations are provided in Data and Models to allow independent assembly of an equivalent corpus.

## GenAI Usage Disclosure

Anthropic Claude Opus 4 has been used to pre-process text samples. Figure 1 was created with Claude Design. Generative AI tools were employed for coding assistance. All AI-written code has been reviewed, tested, and verified by the authors. During the preparation of this manuscript, generative AI tools were employed for editing purposes, including proofreading, grammar correction, vocabulary improvement, and overall language polishing.

## Appendix A Core Clinical symptom descriptions

Natural-language descriptions of depression symptoms, organized by clinical assessment instrument and grouped into three symptom dimensions (mood, somatic, and suicidality). Each entry corresponds to an item or item cluster from the source instrument.

### DSM-5

**Mood.** This person experiences a depressed mood, feeling sad, empty, or hopeless, and may appear tearful to others. They have lost interest or pleasure in activities they used to enjoy. They struggle with feelings of worthlessness or carry excessive or inappropriate guilt that goes beyond normal self-criticism. They also have difficulty thinking clearly, concentrating, or making decisions.

**Somatic.** They experience noticeable changes in weight—either losing or gaining—without intentionally trying to diet, along with shifts in their appetite. Their sleep is disrupted, with difficulty falling or staying asleep, or alternatively, sleeping far more than usual. Others can observe changes in their physical movements and behavior: they may appear restless and agitated, or conversely, slowed down in their speech and actions. They feel persistently tired and lack the energy they once had.

**Suicidality.** The person frequently thinks about death, beyond simply fearing dying. They may have recurring thoughts of suicide without a specific plan, or they may have attempted suicide or developed a specific plan for committing suicide.

### ICD-10

**Mood.** They experience a persistently low mood throughout the day, which others may also observe. They have lost interest or pleasure in activities they would normally enjoy. They struggle with feelings of worthlessness and carry excessive or inappropriate guilt. Their ability to think clearly or concentrate is diminished, and they often feel indecisive. Their self-confidence and self-esteem are reduced.

**Somatic.** They experience changes in their appetite—either eating much less or much more than usual—which leads to noticeable weight loss or gain. Their sleep is disrupted; they may struggle to fall or stay asleep, or conversely find themselves sleeping far more than normal. Even minimal effort leaves them feeling tired and drained, as though their energy has been depleted. Their physical movements and behavior have also changed—they may appear restless and agitated, or alternatively seem slowed down in their speech and actions.

**Suicidality.** They experience recurring thoughts about death or have engaged in suicidal behavior.

### HAM-D (Hamilton Depression Rating Scale)

**Mood**

- **Item 1. Depressed Mood.** They have a gloomy attitude and feel pessimistic about the future. They experience persistent feelings of sadness and may have a tendency to weep.
- **Item 2. Feelings of Guilt.** They engage in self-reproach and feel they have let people down. They experience ideas of guilt, and may believe their present illness is a punishment. In some cases, they may have delusions or hallucinations related to guilt.
- **Item 3. Work and Interests.** They experience feelings of incapacity, listlessness, indecision, and vacillation. They have lost interest in hobbies and show decreased social activities. Their productivity has decreased, and they may be unable to work due to their present illness.
- **Item 4. Anxiety (Psychic).** They experience tension and irritability. They worry about minor matters and have an apprehensive attitude. They may experience fears.

**Somatic**

- **Item 1. Initial Insomnia.** The person has difficulty falling asleep at the beginning of the night.
- **Item 2. Middle Insomnia.** The person feels restless and disturbed during the night, often waking up during sleeping hours.
- **Item 3. Delayed Insomnia.** The person wakes up in the early hours of the morning and is unable to fall asleep again.
- **Item 4. Retardation.** The person experiences slowness in their thought processes, speech, and physical activity. They may appear apathetic or, in more pronounced cases, enter a state of stupor.
- **Item 5. Agitation.** The person displays restlessness that is associated with feelings of anxiety.
- **Item 6. Somatic Anxiety.** The person experiences physical symptoms related to anxiety, which may include gastrointestinal issues such as indigestion, cardiovascular symptoms like palpitations, headaches, respiratory difficulties, or genito-urinary complaints.
- **Item 7. Gastrointestinal Somatic Symptoms.** The person experiences a loss of appetite, a heavy feeling in the abdomen, or constipation.
- **Item 8. General Somatic Symptoms.** The person feels heaviness in their limbs, back, or head, and may experience diffuse backaches, a loss of energy, or easy fatiguability.
- **Item 9. Genital Symptoms.** The person experiences a loss of libido or, in the case of women, menstrual disturbances.
- **Item 10. Hypochondriasis.** The person shows excessive focus on their body and health, ranging from bodily self-absorption to preoccupation with health concerns, a tendency to complain about physical ailments, or fixed false beliefs about having a serious illness.
- **Item 11. Weight Loss.** The person has experienced a noticeable decrease in body weight.

**Suicidality.** The person may experience thoughts related to not wanting to live. This can include feeling that life is not worth living, wishing they were dead, having suicidal ideas or making suicidal gestures, or attempting suicide.

### MADRS (Montgomery–Åsberg Depression Rating Scale)

**Mood**

- **Item 1. Apparent Sadness.** They appear despondent, gloomy, or in despair—beyond ordinary low spirits. This is reflected in their speech, facial expression, and posture. They may look dispirited or unhappy, and at times struggle to brighten up even when circumstances might warrant it.
- **Item 2. Reported Sadness.** They describe feeling depressed, low in spirits, or despondent, regardless of whether this is visible in their appearance. They may report feeling beyond help or without hope. Their mood may be influenced by external circumstances to varying degrees, or it may feel continuous and unvarying.
- **Item 3. Inner Tension.** They experience feelings of ill-defined discomfort, edginess, or inner turmoil. This mental tension can escalate to feelings of panic, dread, or anguish. They may seek reassurance and find it difficult to master these feelings on their own.
- **Item 4. Concentration Difficulties.** They have difficulty collecting their thoughts or sustaining concentration. This may interfere with their ability to read, hold a conversation, or complete tasks that require focused attention.
- **Item 5. Inability to Feel.** They experience reduced interest in their surroundings or in activities that would normally bring them pleasure. Their ability to react with appropriate emotion to circumstances or people feels diminished. They may feel emotionally numb or disconnected from friends and loved ones, struggling to feel anger, grief, or joy.
- **Item 6. Pessimistic Thoughts.** They experience thoughts of guilt, inferiority, self-reproach, or remorse. They may engage in self-accusations or feel increasingly pessimistic about the future, dwelling on ideas of failure, sin, or ruin.

**Somatic**

- **Item 1. Reduced Sleep.** They experience a reduction in the duration or depth of their sleep compared to their usual pattern when feeling well. This may manifest as difficulty falling asleep, lighter or more fitful sleep, sleep that is broken or shortened by several hours, or sleeping only a few hours per night.
- **Item 2. Reduced Appetite.** They experience a diminished appetite compared to when they are well. This presents as a decreased desire for food or a need to force themselves to eat. Food may seem tasteless, and they may require encouragement or persuasion from others to eat at all.
- **Item 3. Lassitude.** They have difficulty getting started with activities and experience slowness in initiating and performing everyday tasks. Routine activities require considerable effort to begin and carry out, and in more pronounced cases, they may feel unable to do anything without assistance from others.

**Suicidality**

- **Item 1.** This person has expressed feelings that life is not worth living and has indicated that a natural death would be welcome. They may feel weary of life, with fleeting or recurring thoughts of suicide.
- **Item 2.** At times, they may believe they would be better off dead, viewing suicide as a possible solution, though not necessarily with specific plans or intention.
- **Item 3.** They experience suicidal thoughts and may have made preparations for suicide. In some cases, they may have explicit plans for suicide or may be actively preparing for it.

### PHQ-9 (Patient Health Questionnaire-9)

**Mood**

- **Item 1.** They have little interest or pleasure in doing things.
- **Item 2.** They feel down, depressed, or hopeless.
- **Item 3.** They feel bad about themselves, or feel like a failure, or feel they have let themselves or their family down.
- **Item 4.** They have trouble concentrating on things, such as reading the newspaper or watching television.

**Somatic**

- **Item 1.** They have trouble falling asleep, staying asleep, or sleep too much.
- **Item 2.** They feel tired or have little energy.
- **Item 3.** They have a poor appetite or overeat.
- **Item 4.** They move or speak so slowly that other people could have noticed, or the opposite—they are so fidgety or restless that they have been moving around a lot more than usual.

**Suicidality.** They have thoughts that they would be better off dead or of hurting themselves in some way.

### PROCEED (Participative Research on Outcomes' and Core Expectations' Elicitation for Depression)

**Mood**

- **Item 1. Perception of Self.** This person often feels that others are against them or out to get them. They experience a profound sense of isolation, as though they are fundamentally alone in the world. They carry a belief that they are a burden to those around them and feel that others do not truly understand them. They struggle with confidence in their own abilities and hold a diminished view of their own worth. At times, they feel disconnected from themselves or their surroundings, and they may have difficulty recognizing themselves—whether in the mirror or in terms of who they feel they are as a person.
- **Item 2. Cognitive Symptoms.** This person finds it difficult to feel motivated or to maintain interest in activities. They believe their condition is untreatable and hold a generally pessimistic view of situations. Their thinking tends to be distorted, often interpreting events in negatively skewed ways. They struggle to make decisions and have difficulty with planning, organizing, and following through on tasks. Adapting to new information or shifting between tasks is challenging for them. They find it hard to solve problems effectively and experience lapses in memory. Maintaining focus and concentration is difficult, as is managing their time. Their perception of time may feel altered—either dragging or slipping away. They have trouble envisioning a positive future and tend to dwell on negative thoughts repeatedly. Their thinking often feels foggy or unclear. They may have reduced capacity to understand or share in others' feelings and show diminished interest in social connection.
- **Item 3. Mood and Emotional Symptoms.** This person experiences significant psychological pain and a pervasive sense that things will not improve. They feel powerless to change their circumstances and see themselves as having little value. A general sense of feeling bad pervades their experience, often accompanied by guilt. Sadness is a frequent companion, along with feelings of anxiety and a fear of failing at things they attempt. They often feel restless and on edge, becoming easily irritated and frustrated. They have difficulty managing their emotions and controlling their impulses. Their mood may not shift appropriately in response to positive events. They have lost their sense of humor and find little or no pleasure in activities they once enjoyed. They often feel empty inside, and their emotional responses in general feel muted or dulled.

**Somatic**

- **Item 1. Physical Symptoms.** They experience pain or other physical symptoms. Their appetite or weight has changed. They have difficulty with sexual function.
- **Item 2. Sleep.** They sleep excessively. Their sleep is disturbed or restless. They have difficulty falling or staying asleep.
- **Item 3. Psychomotor and Expression.** They cry. They speak very little or not at all. Their facial expressions appear flat or diminished. Their movements and reactions are slowed.
- **Item 4. Energy.** They feel that everything requires effort. They feel physically weak. They have low energy. They feel fatigued or tired.

**Suicidality.** **Autoaggression symptoms.** The individual experiences thoughts related to their desire to continue living. They report suicidal ideation and engage in self-harming behaviors. They have attempted suicide.

## Appendix B PERMANOVA and PERMDISP statistics at the gated peaks

The centroid metrics peaked at the operating layer more often than Earth Mover's and Energy Distance did (25 of their 40 gated peaks, against 11 of 40). Gated peaks at other layers also sat closer to the dispersion threshold, with PERMDISP $$p$$ below 0.10 for 52% of them against 31% at the operating layer.

<figure id="tab-permanova" style="text-align: center; margin: 20px 0;">
  <p style="margin-bottom: 0.3em;"><b>Table B1:</b> PERMANOVA and PERMDISP statistics at each metric's gated peak, with <i>p</i>-values beneath each statistic (9,999 permutations; 0.0001 is the smallest attainable <i>p</i>).</p>
<div class="model-views" style="overflow-x: auto;">
<div data-model="Gemma-3-27B-PT">
<p style="margin: 0 0 0.3em; font-size: 0.9em; color: #555;">Model: Gemma-3-27B-PT · L21</p>
<table style="width: 100%;">
<thead><tr><th style="text-align: left;">Metric</th><th style="text-align: center;">Layer</th><th style="text-align: center;">PERMANOVA<br>pseudo-<i>F</i></th><th style="text-align: center;">PERMDISP<br><i>F</i></th></tr></thead>
<tbody>
<tr><td style="text-align: left;">Centroid Cosine<br><span style="color: #555;">raw</span></td><td style="text-align: center;">37</td><td style="text-align: center;">6.39<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">4.28<br><span style="color: #555;"><i>p</i> = 0.0518</span></td></tr>
<tr><td style="text-align: left;">Centroid Cosine<br><span style="color: #555;">unit-normalized</span></td><td style="text-align: center;">40</td><td style="text-align: center;">6.37<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">3.76<br><span style="color: #555;"><i>p</i> = 0.0843</span></td></tr>
<tr><td style="text-align: left;">Centroid Euclidean<br><span style="color: #555;">raw</span></td><td style="text-align: center;">21</td><td style="text-align: center;">6.91<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">1.24<br><span style="color: #555;"><i>p</i> = 0.4319</span></td></tr>
<tr><td style="text-align: left;">Centroid Euclidean<br><span style="color: #555;">unit-normalized</span></td><td style="text-align: center;">35</td><td style="text-align: center;">3.66<br><span style="color: #555;"><i>p</i> = 0.0004</span></td><td style="text-align: center;">3.83<br><span style="color: #555;"><i>p</i> = 0.1095</span></td></tr>
<tr><td style="text-align: left;">Earth Mover&#x27;s<br><span style="color: #555;">raw</span></td><td style="text-align: center;">21</td><td style="text-align: center;">2.83<br><span style="color: #555;"><i>p</i> = 0.0002</span></td><td style="text-align: center;">1.20<br><span style="color: #555;"><i>p</i> = 0.4698</span></td></tr>
<tr><td style="text-align: left;">Earth Mover&#x27;s<br><span style="color: #555;">unit-normalized</span></td><td style="text-align: center;">35</td><td style="text-align: center;">2.47<br><span style="color: #555;"><i>p</i> = 0.0021</span></td><td style="text-align: center;">2.92<br><span style="color: #555;"><i>p</i> = 0.1867</span></td></tr>
<tr><td style="text-align: left;">Energy<br><span style="color: #555;">raw</span></td><td style="text-align: center;">21</td><td style="text-align: center;">6.48<br><span style="color: #555;"><i>p</i> = 0.0028</span></td><td style="text-align: center;">0.68<br><span style="color: #555;"><i>p</i> = 0.6419</span></td></tr>
<tr><td style="text-align: left;">Energy<br><span style="color: #555;">unit-normalized</span></td><td style="text-align: center;">35</td><td style="text-align: center;">5.77<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">4.58<br><span style="color: #555;"><i>p</i> = 0.0579</span></td></tr>
</tbody>
</table>
</div>
<div data-model="Gemma-3-27B-IT" hidden>
<p style="margin: 0 0 0.3em; font-size: 0.9em; color: #555;">Model: Gemma-3-27B-IT · L21</p>
<table style="width: 100%;">
<thead><tr><th style="text-align: left;">Metric</th><th style="text-align: center;">Layer</th><th style="text-align: center;">PERMANOVA<br>pseudo-<i>F</i></th><th style="text-align: center;">PERMDISP<br><i>F</i></th></tr></thead>
<tbody>
<tr><td style="text-align: left;">Centroid Cosine<br><span style="color: #555;">raw</span></td><td style="text-align: center;">21</td><td style="text-align: center;">6.37<br><span style="color: #555;"><i>p</i> = 0.0130</span></td><td style="text-align: center;">0.59<br><span style="color: #555;"><i>p</i> = 0.6232</span></td></tr>
<tr><td style="text-align: left;">Centroid Cosine<br><span style="color: #555;">unit-normalized</span></td><td style="text-align: center;">21</td><td style="text-align: center;">5.04<br><span style="color: #555;"><i>p</i> = 0.0313</span></td><td style="text-align: center;">0.61<br><span style="color: #555;"><i>p</i> = 0.6157</span></td></tr>
<tr><td style="text-align: left;">Centroid Euclidean<br><span style="color: #555;">raw</span></td><td style="text-align: center;">26</td><td style="text-align: center;">8.93<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">1.54<br><span style="color: #555;"><i>p</i> = 0.2785</span></td></tr>
<tr><td style="text-align: left;">Centroid Euclidean<br><span style="color: #555;">unit-normalized</span></td><td style="text-align: center;">21</td><td style="text-align: center;">4.16<br><span style="color: #555;"><i>p</i> = 0.0023</span></td><td style="text-align: center;">1.43<br><span style="color: #555;"><i>p</i> = 0.3669</span></td></tr>
<tr><td style="text-align: left;">Earth Mover&#x27;s<br><span style="color: #555;">raw</span></td><td style="text-align: center;">31</td><td style="text-align: center;">3.17<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">4.78<br><span style="color: #555;"><i>p</i> = 0.0536</span></td></tr>
<tr><td style="text-align: left;">Earth Mover&#x27;s<br><span style="color: #555;">unit-normalized</span></td><td style="text-align: center;">21</td><td style="text-align: center;">2.51<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">4.53<br><span style="color: #555;"><i>p</i> = 0.0717</span></td></tr>
<tr><td style="text-align: left;">Energy<br><span style="color: #555;">raw</span></td><td style="text-align: center;">31</td><td style="text-align: center;">9.77<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">2.07<br><span style="color: #555;"><i>p</i> = 0.1536</span></td></tr>
<tr><td style="text-align: left;">Energy<br><span style="color: #555;">unit-normalized</span></td><td style="text-align: center;">21</td><td style="text-align: center;">6.23<br><span style="color: #555;"><i>p</i> = 0.0017</span></td><td style="text-align: center;">0.77<br><span style="color: #555;"><i>p</i> = 0.5648</span></td></tr>
</tbody>
</table>
</div>
<div data-model="Gemma-3-12B-PT" hidden>
<p style="margin: 0 0 0.3em; font-size: 0.9em; color: #555;">Model: Gemma-3-12B-PT · L13</p>
<table style="width: 100%;">
<thead><tr><th style="text-align: left;">Metric</th><th style="text-align: center;">Layer</th><th style="text-align: center;">PERMANOVA<br>pseudo-<i>F</i></th><th style="text-align: center;">PERMDISP<br><i>F</i></th></tr></thead>
<tbody>
<tr><td style="text-align: left;">Centroid Cosine<br><span style="color: #555;">raw</span></td><td style="text-align: center;">13</td><td style="text-align: center;">5.91<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">2.63<br><span style="color: #555;"><i>p</i> = 0.1571</span></td></tr>
<tr><td style="text-align: left;">Centroid Cosine<br><span style="color: #555;">unit-normalized</span></td><td style="text-align: center;">13</td><td style="text-align: center;">5.60<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">2.37<br><span style="color: #555;"><i>p</i> = 0.2047</span></td></tr>
<tr><td style="text-align: left;">Centroid Euclidean<br><span style="color: #555;">raw</span></td><td style="text-align: center;">16</td><td style="text-align: center;">3.73<br><span style="color: #555;"><i>p</i> = 0.0063</span></td><td style="text-align: center;">1.19<br><span style="color: #555;"><i>p</i> = 0.3793</span></td></tr>
<tr><td style="text-align: left;">Centroid Euclidean<br><span style="color: #555;">unit-normalized</span></td><td style="text-align: center;">13</td><td style="text-align: center;">3.77<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">3.41<br><span style="color: #555;"><i>p</i> = 0.1105</span></td></tr>
<tr><td style="text-align: left;">Earth Mover&#x27;s<br><span style="color: #555;">raw</span></td><td style="text-align: center;">21</td><td style="text-align: center;">2.43<br><span style="color: #555;"><i>p</i> = 0.0185</span></td><td style="text-align: center;">0.38<br><span style="color: #555;"><i>p</i> = 0.7998</span></td></tr>
<tr><td style="text-align: left;">Earth Mover&#x27;s<br><span style="color: #555;">unit-normalized</span></td><td style="text-align: center;">37</td><td style="text-align: center;">2.48<br><span style="color: #555;"><i>p</i> = 0.0003</span></td><td style="text-align: center;">4.16<br><span style="color: #555;"><i>p</i> = 0.0811</span></td></tr>
<tr><td style="text-align: left;">Energy<br><span style="color: #555;">raw</span></td><td style="text-align: center;">47</td><td style="text-align: center;">5.43<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">1.92<br><span style="color: #555;"><i>p</i> = 0.2039</span></td></tr>
<tr><td style="text-align: left;">Energy<br><span style="color: #555;">unit-normalized</span></td><td style="text-align: center;">27</td><td style="text-align: center;">4.65<br><span style="color: #555;"><i>p</i> = 0.0003</span></td><td style="text-align: center;">3.37<br><span style="color: #555;"><i>p</i> = 0.1486</span></td></tr>
</tbody>
</table>
</div>
<div data-model="Gemma-3-12B-IT" hidden>
<p style="margin: 0 0 0.3em; font-size: 0.9em; color: #555;">Model: Gemma-3-12B-IT · L14</p>
<table style="width: 100%;">
<thead><tr><th style="text-align: left;">Metric</th><th style="text-align: center;">Layer</th><th style="text-align: center;">PERMANOVA<br>pseudo-<i>F</i></th><th style="text-align: center;">PERMDISP<br><i>F</i></th></tr></thead>
<tbody>
<tr><td style="text-align: left;">Centroid Cosine<br><span style="color: #555;">raw</span></td><td style="text-align: center;">14</td><td style="text-align: center;">6.49<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">4.08<br><span style="color: #555;"><i>p</i> = 0.0567</span></td></tr>
<tr><td style="text-align: left;">Centroid Cosine<br><span style="color: #555;">unit-normalized</span></td><td style="text-align: center;">35</td><td style="text-align: center;">6.52<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">4.31<br><span style="color: #555;"><i>p</i> = 0.0602</span></td></tr>
<tr><td style="text-align: left;">Centroid Euclidean<br><span style="color: #555;">raw</span></td><td style="text-align: center;">14</td><td style="text-align: center;">8.03<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">1.42<br><span style="color: #555;"><i>p</i> = 0.2850</span></td></tr>
<tr><td style="text-align: left;">Centroid Euclidean<br><span style="color: #555;">unit-normalized</span></td><td style="text-align: center;">30</td><td style="text-align: center;">3.92<br><span style="color: #555;"><i>p</i> = 0.0002</span></td><td style="text-align: center;">3.95<br><span style="color: #555;"><i>p</i> = 0.1005</span></td></tr>
<tr><td style="text-align: left;">Earth Mover&#x27;s<br><span style="color: #555;">raw</span></td><td style="text-align: center;">21</td><td style="text-align: center;">3.26<br><span style="color: #555;"><i>p</i> = 0.0014</span></td><td style="text-align: center;">0.01<br><span style="color: #555;"><i>p</i> = 0.9965</span></td></tr>
<tr><td style="text-align: left;">Earth Mover&#x27;s<br><span style="color: #555;">unit-normalized</span></td><td style="text-align: center;">35</td><td style="text-align: center;">2.55<br><span style="color: #555;"><i>p</i> = 0.0002</span></td><td style="text-align: center;">3.83<br><span style="color: #555;"><i>p</i> = 0.1091</span></td></tr>
<tr><td style="text-align: left;">Energy<br><span style="color: #555;">raw</span></td><td style="text-align: center;">19</td><td style="text-align: center;">9.58<br><span style="color: #555;"><i>p</i> = 0.0002</span></td><td style="text-align: center;">1.72<br><span style="color: #555;"><i>p</i> = 0.2243</span></td></tr>
<tr><td style="text-align: left;">Energy<br><span style="color: #555;">unit-normalized</span></td><td style="text-align: center;">13</td><td style="text-align: center;">4.93<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">3.22<br><span style="color: #555;"><i>p</i> = 0.0960</span></td></tr>
</tbody>
</table>
</div>
<div data-model="Gemma-3-4B-PT" hidden>
<p style="margin: 0 0 0.3em; font-size: 0.9em; color: #555;">Model: Gemma-3-4B-PT · L20</p>
<table style="width: 100%;">
<thead><tr><th style="text-align: left;">Metric</th><th style="text-align: center;">Layer</th><th style="text-align: center;">PERMANOVA<br>pseudo-<i>F</i></th><th style="text-align: center;">PERMDISP<br><i>F</i></th></tr></thead>
<tbody>
<tr><td style="text-align: left;">Centroid Cosine<br><span style="color: #555;">raw</span></td><td style="text-align: center;">17</td><td style="text-align: center;">5.81<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">4.25<br><span style="color: #555;"><i>p</i> = 0.0705</span></td></tr>
<tr><td style="text-align: left;">Centroid Cosine<br><span style="color: #555;">unit-normalized</span></td><td style="text-align: center;">23</td><td style="text-align: center;">6.65<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">3.24<br><span style="color: #555;"><i>p</i> = 0.1368</span></td></tr>
<tr><td style="text-align: left;">Centroid Euclidean<br><span style="color: #555;">raw</span></td><td style="text-align: center;">9</td><td style="text-align: center;">4.78<br><span style="color: #555;"><i>p</i> = 0.0011</span></td><td style="text-align: center;">0.26<br><span style="color: #555;"><i>p</i> = 0.8307</span></td></tr>
<tr><td style="text-align: left;">Centroid Euclidean<br><span style="color: #555;">unit-normalized</span></td><td style="text-align: center;">20</td><td style="text-align: center;">3.83<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">3.45<br><span style="color: #555;"><i>p</i> = 0.1328</span></td></tr>
<tr><td style="text-align: left;">Earth Mover&#x27;s<br><span style="color: #555;">raw</span></td><td style="text-align: center;">18</td><td style="text-align: center;">2.52<br><span style="color: #555;"><i>p</i> = 0.0016</span></td><td style="text-align: center;">2.12<br><span style="color: #555;"><i>p</i> = 0.2330</span></td></tr>
<tr><td style="text-align: left;">Earth Mover&#x27;s<br><span style="color: #555;">unit-normalized</span></td><td style="text-align: center;">24</td><td style="text-align: center;">2.57<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">5.91<br><span style="color: #555;"><i>p</i> = 0.0555</span></td></tr>
<tr><td style="text-align: left;">Energy<br><span style="color: #555;">raw</span></td><td style="text-align: center;">20</td><td style="text-align: center;">6.55<br><span style="color: #555;"><i>p</i> = 0.0008</span></td><td style="text-align: center;">0.67<br><span style="color: #555;"><i>p</i> = 0.5486</span></td></tr>
<tr><td style="text-align: left;">Energy<br><span style="color: #555;">unit-normalized</span></td><td style="text-align: center;">20</td><td style="text-align: center;">5.87<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">3.66<br><span style="color: #555;"><i>p</i> = 0.0935</span></td></tr>
</tbody>
</table>
</div>
<div data-model="Gemma-3-4B-IT" hidden>
<p style="margin: 0 0 0.3em; font-size: 0.9em; color: #555;">Model: Gemma-3-4B-IT · L20</p>
<table style="width: 100%;">
<thead><tr><th style="text-align: left;">Metric</th><th style="text-align: center;">Layer</th><th style="text-align: center;">PERMANOVA<br>pseudo-<i>F</i></th><th style="text-align: center;">PERMDISP<br><i>F</i></th></tr></thead>
<tbody>
<tr><td style="text-align: left;">Centroid Cosine<br><span style="color: #555;">raw</span></td><td style="text-align: center;">17</td><td style="text-align: center;">6.12<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">4.18<br><span style="color: #555;"><i>p</i> = 0.0734</span></td></tr>
<tr><td style="text-align: left;">Centroid Cosine<br><span style="color: #555;">unit-normalized</span></td><td style="text-align: center;">23</td><td style="text-align: center;">6.83<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">2.82<br><span style="color: #555;"><i>p</i> = 0.1610</span></td></tr>
<tr><td style="text-align: left;">Centroid Euclidean<br><span style="color: #555;">raw</span></td><td style="text-align: center;">20</td><td style="text-align: center;">5.59<br><span style="color: #555;"><i>p</i> = 0.0008</span></td><td style="text-align: center;">0.46<br><span style="color: #555;"><i>p</i> = 0.6553</span></td></tr>
<tr><td style="text-align: left;">Centroid Euclidean<br><span style="color: #555;">unit-normalized</span></td><td style="text-align: center;">20</td><td style="text-align: center;">3.57<br><span style="color: #555;"><i>p</i> = 0.0011</span></td><td style="text-align: center;">2.40<br><span style="color: #555;"><i>p</i> = 0.2450</span></td></tr>
<tr><td style="text-align: left;">Earth Mover&#x27;s<br><span style="color: #555;">raw</span></td><td style="text-align: center;">18</td><td style="text-align: center;">2.87<br><span style="color: #555;"><i>p</i> = 0.0008</span></td><td style="text-align: center;">0.86<br><span style="color: #555;"><i>p</i> = 0.5358</span></td></tr>
<tr><td style="text-align: left;">Earth Mover&#x27;s<br><span style="color: #555;">unit-normalized</span></td><td style="text-align: center;">34</td><td style="text-align: center;">2.74<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">3.19<br><span style="color: #555;"><i>p</i> = 0.1201</span></td></tr>
<tr><td style="text-align: left;">Energy<br><span style="color: #555;">raw</span></td><td style="text-align: center;">20</td><td style="text-align: center;">8.45<br><span style="color: #555;"><i>p</i> = 0.0002</span></td><td style="text-align: center;">0.15<br><span style="color: #555;"><i>p</i> = 0.8790</span></td></tr>
<tr><td style="text-align: left;">Energy<br><span style="color: #555;">unit-normalized</span></td><td style="text-align: center;">20</td><td style="text-align: center;">5.89<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">3.57<br><span style="color: #555;"><i>p</i> = 0.1048</span></td></tr>
</tbody>
</table>
</div>
<div data-model="Mistral-24B-PT" hidden>
<p style="margin: 0 0 0.3em; font-size: 0.9em; color: #555;">Model: Mistral-24B-PT · L8</p>
<table style="width: 100%;">
<thead><tr><th style="text-align: left;">Metric</th><th style="text-align: center;">Layer</th><th style="text-align: center;">PERMANOVA<br>pseudo-<i>F</i></th><th style="text-align: center;">PERMDISP<br><i>F</i></th></tr></thead>
<tbody>
<tr><td style="text-align: left;">Centroid Cosine<br><span style="color: #555;">raw</span></td><td style="text-align: center;">8</td><td style="text-align: center;">5.84<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">3.85<br><span style="color: #555;"><i>p</i> = 0.0818</span></td></tr>
<tr><td style="text-align: left;">Centroid Cosine<br><span style="color: #555;">unit-normalized</span></td><td style="text-align: center;">8</td><td style="text-align: center;">5.80<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">3.48<br><span style="color: #555;"><i>p</i> = 0.0974</span></td></tr>
<tr><td style="text-align: left;">Centroid Euclidean<br><span style="color: #555;">raw</span></td><td style="text-align: center;">8</td><td style="text-align: center;">3.49<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">3.38<br><span style="color: #555;"><i>p</i> = 0.1106</span></td></tr>
<tr><td style="text-align: left;">Centroid Euclidean<br><span style="color: #555;">unit-normalized</span></td><td style="text-align: center;">8</td><td style="text-align: center;">3.50<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">3.58<br><span style="color: #555;"><i>p</i> = 0.0963</span></td></tr>
<tr><td style="text-align: left;">Earth Mover&#x27;s<br><span style="color: #555;">raw</span></td><td style="text-align: center;">13</td><td style="text-align: center;">2.34<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">7.54<br><span style="color: #555;"><i>p</i> = 0.0514</span></td></tr>
<tr><td style="text-align: left;">Earth Mover&#x27;s<br><span style="color: #555;">unit-normalized</span></td><td style="text-align: center;">8</td><td style="text-align: center;">2.26<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">10.65<br><span style="color: #555;"><i>p</i> = 0.1052</span></td></tr>
<tr><td style="text-align: left;">Energy<br><span style="color: #555;">raw</span></td><td style="text-align: center;">11</td><td style="text-align: center;">5.60<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">3.83<br><span style="color: #555;"><i>p</i> = 0.0666</span></td></tr>
<tr><td style="text-align: left;">Energy<br><span style="color: #555;">unit-normalized</span></td><td style="text-align: center;">11</td><td style="text-align: center;">5.76<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">3.91<br><span style="color: #555;"><i>p</i> = 0.0571</span></td></tr>
</tbody>
</table>
</div>
<div data-model="Mistral-24B-IT" hidden>
<p style="margin: 0 0 0.3em; font-size: 0.9em; color: #555;">Model: Mistral-24B-IT · L8</p>
<table style="width: 100%;">
<thead><tr><th style="text-align: left;">Metric</th><th style="text-align: center;">Layer</th><th style="text-align: center;">PERMANOVA<br>pseudo-<i>F</i></th><th style="text-align: center;">PERMDISP<br><i>F</i></th></tr></thead>
<tbody>
<tr><td style="text-align: left;">Centroid Cosine<br><span style="color: #555;">raw</span></td><td style="text-align: center;">8</td><td style="text-align: center;">6.16<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">4.20<br><span style="color: #555;"><i>p</i> = 0.0693</span></td></tr>
<tr><td style="text-align: left;">Centroid Cosine<br><span style="color: #555;">unit-normalized</span></td><td style="text-align: center;">8</td><td style="text-align: center;">6.09<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">3.74<br><span style="color: #555;"><i>p</i> = 0.0839</span></td></tr>
<tr><td style="text-align: left;">Centroid Euclidean<br><span style="color: #555;">raw</span></td><td style="text-align: center;">8</td><td style="text-align: center;">3.67<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">3.46<br><span style="color: #555;"><i>p</i> = 0.1037</span></td></tr>
<tr><td style="text-align: left;">Centroid Euclidean<br><span style="color: #555;">unit-normalized</span></td><td style="text-align: center;">8</td><td style="text-align: center;">3.67<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">3.81<br><span style="color: #555;"><i>p</i> = 0.0825</span></td></tr>
<tr><td style="text-align: left;">Earth Mover&#x27;s<br><span style="color: #555;">raw</span></td><td style="text-align: center;">12</td><td style="text-align: center;">2.47<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">7.07<br><span style="color: #555;"><i>p</i> = 0.0539</span></td></tr>
<tr><td style="text-align: left;">Earth Mover&#x27;s<br><span style="color: #555;">unit-normalized</span></td><td style="text-align: center;">8</td><td style="text-align: center;">2.30<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">10.82<br><span style="color: #555;"><i>p</i> = 0.0974</span></td></tr>
<tr><td style="text-align: left;">Energy<br><span style="color: #555;">raw</span></td><td style="text-align: center;">11</td><td style="text-align: center;">5.89<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">4.15<br><span style="color: #555;"><i>p</i> = 0.0555</span></td></tr>
<tr><td style="text-align: left;">Energy<br><span style="color: #555;">unit-normalized</span></td><td style="text-align: center;">13</td><td style="text-align: center;">5.90<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">4.16<br><span style="color: #555;"><i>p</i> = 0.0513</span></td></tr>
</tbody>
</table>
</div>
<div data-model="OLMo-2-32B-PT" hidden>
<p style="margin: 0 0 0.3em; font-size: 0.9em; color: #555;">Model: OLMo-2-32B-PT · L16</p>
<table style="width: 100%;">
<thead><tr><th style="text-align: left;">Metric</th><th style="text-align: center;">Layer</th><th style="text-align: center;">PERMANOVA<br>pseudo-<i>F</i></th><th style="text-align: center;">PERMDISP<br><i>F</i></th></tr></thead>
<tbody>
<tr><td style="text-align: left;">Centroid Cosine<br><span style="color: #555;">raw</span></td><td style="text-align: center;">16</td><td style="text-align: center;">5.41<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">3.31<br><span style="color: #555;"><i>p</i> = 0.1153</span></td></tr>
<tr><td style="text-align: left;">Centroid Cosine<br><span style="color: #555;">unit-normalized</span></td><td style="text-align: center;">16</td><td style="text-align: center;">5.23<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">3.05<br><span style="color: #555;"><i>p</i> = 0.1391</span></td></tr>
<tr><td style="text-align: left;">Centroid Euclidean<br><span style="color: #555;">raw</span></td><td style="text-align: center;">15</td><td style="text-align: center;">3.25<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">3.99<br><span style="color: #555;"><i>p</i> = 0.0631</span></td></tr>
<tr><td style="text-align: left;">Centroid Euclidean<br><span style="color: #555;">unit-normalized</span></td><td style="text-align: center;">15</td><td style="text-align: center;">3.19<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">2.61<br><span style="color: #555;"><i>p</i> = 0.1763</span></td></tr>
<tr><td style="text-align: left;">Earth Mover&#x27;s<br><span style="color: #555;">raw</span></td><td style="text-align: center;">24</td><td style="text-align: center;">1.98<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">8.98<br><span style="color: #555;"><i>p</i> = 0.1007</span></td></tr>
<tr><td style="text-align: left;">Earth Mover&#x27;s<br><span style="color: #555;">unit-normalized</span></td><td style="text-align: center;">16</td><td style="text-align: center;">2.08<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">13.59<br><span style="color: #555;"><i>p</i> = 0.0813</span></td></tr>
<tr><td style="text-align: left;">Energy<br><span style="color: #555;">raw</span></td><td style="text-align: center;">26</td><td style="text-align: center;">4.36<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">3.57<br><span style="color: #555;"><i>p</i> = 0.0812</span></td></tr>
<tr><td style="text-align: left;">Energy<br><span style="color: #555;">unit-normalized</span></td><td style="text-align: center;">42</td><td style="text-align: center;">5.03<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">4.31<br><span style="color: #555;"><i>p</i> = 0.0545</span></td></tr>
</tbody>
</table>
</div>
<div data-model="OLMo-2-32B-IT" hidden>
<p style="margin: 0 0 0.3em; font-size: 0.9em; color: #555;">Model: OLMo-2-32B-IT · L16</p>
<table style="width: 100%;">
<thead><tr><th style="text-align: left;">Metric</th><th style="text-align: center;">Layer</th><th style="text-align: center;">PERMANOVA<br>pseudo-<i>F</i></th><th style="text-align: center;">PERMDISP<br><i>F</i></th></tr></thead>
<tbody>
<tr><td style="text-align: left;">Centroid Cosine<br><span style="color: #555;">raw</span></td><td style="text-align: center;">16</td><td style="text-align: center;">5.46<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">3.34<br><span style="color: #555;"><i>p</i> = 0.1112</span></td></tr>
<tr><td style="text-align: left;">Centroid Cosine<br><span style="color: #555;">unit-normalized</span></td><td style="text-align: center;">16</td><td style="text-align: center;">5.29<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">2.98<br><span style="color: #555;"><i>p</i> = 0.1459</span></td></tr>
<tr><td style="text-align: left;">Centroid Euclidean<br><span style="color: #555;">raw</span></td><td style="text-align: center;">16</td><td style="text-align: center;">3.23<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">2.90<br><span style="color: #555;"><i>p</i> = 0.1365</span></td></tr>
<tr><td style="text-align: left;">Centroid Euclidean<br><span style="color: #555;">unit-normalized</span></td><td style="text-align: center;">15</td><td style="text-align: center;">3.23<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">2.78<br><span style="color: #555;"><i>p</i> = 0.1593</span></td></tr>
<tr><td style="text-align: left;">Earth Mover&#x27;s<br><span style="color: #555;">raw</span></td><td style="text-align: center;">24</td><td style="text-align: center;">2.00<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">8.86<br><span style="color: #555;"><i>p</i> = 0.0967</span></td></tr>
<tr><td style="text-align: left;">Earth Mover&#x27;s<br><span style="color: #555;">unit-normalized</span></td><td style="text-align: center;">15</td><td style="text-align: center;">2.12<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">15.63<br><span style="color: #555;"><i>p</i> = 0.0569</span></td></tr>
<tr><td style="text-align: left;">Energy<br><span style="color: #555;">raw</span></td><td style="text-align: center;">26</td><td style="text-align: center;">4.46<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">4.28<br><span style="color: #555;"><i>p</i> = 0.0504</span></td></tr>
<tr><td style="text-align: left;">Energy<br><span style="color: #555;">unit-normalized</span></td><td style="text-align: center;">64</td><td style="text-align: center;">5.59<br><span style="color: #555;"><i>p</i> ≤ 0.0001</span></td><td style="text-align: center;">3.96<br><span style="color: #555;"><i>p</i> = 0.0660</span></td></tr>
</tbody>
</table>
</div>
</div>
</figure>

## Appendix C Within-source clinician agreement

Because ReDSM5 supplies most somatic and suicidality excerpts but no mood excerpts (Table 1), symptom group and source are associated in the Naturalistic corpus, so we repeated the analysis within each source. On the book-sourced excerpts alone ($$n = 229$$), each group had the highest median coefficient on its own axis in all ten models, as it did on the *Handbook* excerpts alone ($$n = 172$$); on *Darkness Visible* alone ($$n = 57$$), the ordering held in every model except Gemma-3-27B-PT, where only one of the three groups ranked first on its own axis. Within ReDSM5 ($$n = 194$$), somatic excerpts scored above suicidality excerpts on the somatic axis (AUC 0.809–0.957), and suicidality excerpts above somatic excerpts on the suicidality axis (AUC 0.733–0.994; one-sided Mann–Whitney $$U$$, all $$p < 10^{-7}$$), in every model. The two Gemma-3-27B models had the lowest suicidality-axis AUCs (0.737 pre-trained, 0.733 instruction-tuned), against 0.986–0.994 in the other eight. Source shifts the level of the coefficients (ReDSM5 suicidality excerpts scored higher on the suicidality axis than book-sourced ones in every model), while the ordering held within each source in every model except Gemma-3-27B-PT on *Darkness Visible*.
