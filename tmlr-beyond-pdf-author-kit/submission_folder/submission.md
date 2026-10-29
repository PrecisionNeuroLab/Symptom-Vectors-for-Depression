---
layout: distill
title: Interpretable Symptom Vectors for Depression in a Large Language Model
description: Depression symptom categories are geometrically separable in the residual streams of ten open-weight language models, and projecting held-out text onto clinician-derived Symptom Vectors recovers clinician annotations in every one.
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
  - name: Methods
    subsections:
      - name: Data and preprocessing
      - name: Per-layer separability
      - name: Semantic projection for symptom identification
      - name: Depressive and positive affect distinction
  - name: Results
    subsections:
      - name: Symptom groups are separable in all ten models
      - name: Semantic projection coefficients align with clinician ratings
      - name: A single Depression Vector distinguishes depressive and non-depressive text
      - name: Summary across models
      - name: Pre-trained versus instruction-tuned models
  - name: Discussion
  - name: Limitations
  - name: Conclusion
  - name: Broader Impact Statement
  - name: Reproducibility Statement
  - name: GenAI Usage Disclosure
  - name: Appendix A Core Clinical symptom descriptions
---

## Abstract

Patients with depression present with diverse symptom profiles, yet clinical practice routinely reduces this variation to a single severity score. Large language models (LLMs) can potentially capture various symptoms and their severity from patient speech. However, how depressive symptoms are represented inside LLMs remains poorly understood, limiting clinical trust. To examine whether internal model activations match clinician judgment, we analyzed the residual streams of ten open-weight pre-trained and instruction-tuned models from three families (Gemma-3, Mistral-Small, OLMo-2) using mechanistic interpretability techniques. Recording activations across symptom descriptions drawn from validated clinical instruments, we found that symptom groups were geometrically separable in every model, with the most separable layer at 20–34% of network depth in eight of the ten. Using Semantic Projection, we then projected held-out naturalistic text onto Symptom Vectors constructed from these instruments at each model's most separable layer. The resulting per-symptom coefficients preserved clinician-annotated rank ordering across mood, somatic, and suicidality axes in all ten models. Furthermore, a single Depression Vector separates held-out depressive from non-depressive text in every model (AUC = 0.730–0.965), which can be used as an emotional valence gate that restricts symptom projection to depressive speech. These results reveal a decorrelated, clinician-aligned symptom signal readable directly from internal activations across model families, suggesting the pipeline can serve as an interpretability screen when choosing open models for clinical NLP research and offering a foundation for interpretable depression-assessment tools.

<p style="font-size: 0.9em; color: #555;"><b>Reading the interactive figures.</b> Every figure except Figure 1 is interactive: hover a point to read the excerpt it represents and its exact value, drag to zoom, double-click to reset, and click a legend entry to hide or isolate a group. Figure 5 can be rotated. Figures 3–5 and Tables 3–4 show one model at a time; the floating Model selector at the upper right switches all of them to another model at once. Figure 6 shows every model side by side.</p>

## Introduction

Major depressive disorder spans highly diverse symptom profiles, biological underpinnings, and treatment trajectories, a multidimensional complexity that remains one of the most significant barriers to effective psychiatric care. In the STAR\*D cohort, for example, 3,703 patients meeting the same DSM-5 criteria produced 1,030 unique symptom profiles <d-cite key="fried2015depression"></d-cite>. Standard questionnaires like PHQ and HAM-D prioritize brevity; they collapse rich symptom dimensions into a single score while overlooking symptoms that matter to patients <d-cite key="chevance2020identifying"></d-cite>. These discarded nuances, however, are preserved when patients describe their own experiences in naturalistic language. Symptom-level extraction from such language could help differentiate patients with divergent underlying profiles. Thus, symptom extraction from natural language has been a long-standing goal in clinical NLP.

Recent LLMs, which can take context into account and process text as a whole, have renewed interest in this goal. Existing systems, however, read their assessments from generated output, which yields broad labels rather than symptom-level signals and is an unreliable window into model internals (see Related Work). Mechanistic interpretability techniques offer a route to quantitative, per-symptom signals read directly from the model's internal representations. To our knowledge, no prior work has mapped a clinically validated depression symptom taxonomy onto decorrelated per-symptom signals in an LLM's internal activations.

As an LLM processes text, it builds internal representations in a high-dimensional activation space known as the residual stream. Prior work suggests emotions are encoded as linear directions in this space; depression symptom categories may be similarly represented. Applying mechanistic interpretability techniques to ten pre-trained and instruction-tuned models from three families, we show that these categories (mood/emotional/cognitive, somatic, and suicidality) are geometrically separable in every model's activations, at a comparable relative depth in most. In every model, projecting held-out naturalistic narratives onto clinician-derived Symptom Vectors yields coefficients that predict clinician category annotations, and a single Depression Vector distinguishes depressive from positive affect text. This study establishes a pipeline for extracting interpretable symptom signals directly from an LLM's internal state, preserving the nuanced profiles that brief severity scores obscure.

## Related Work

**LLMs for depression assessment.** Existing LLM-based systems for mental health typically produce broad diagnostic labels or sum scores rather than symptom-level assessments <d-cite key="yang2024mentallama,xu2024mentalllm,lamichhane2023evaluation"></d-cite>, or require task-specific fine-tuning for symptom-level analysis <d-cite key="weber2025finetuned"></d-cite>. Generated outputs are also an unreliable window into model internals: GPT-4's internal schema of depression underemphasizes suicidality and overemphasizes psychomotor symptoms relative to expert and self-report ratings <d-cite key="ganesan2026explaining"></d-cite>. We instead read symptom signals from the internal activations of an unmodified pre-trained model.

**Linear representations of concepts and affect.** Semantic categories are encoded as approximately linear directions in LLM residual streams <d-cite key="park2025geometry"></d-cite>. Emotion representations follow the same pattern, with sentiment localized to specific layers in LLaMA <d-cite key="hollinsworth2024sentiment,dipalma2025llamas"></d-cite>, and the emotional information models pick up from text can be traced to specific, interpretable internal signals <d-cite key="tak2025mechanistic"></d-cite>. This work characterizes coarse valence; we ask whether a finer, instrument-derived symptom taxonomy is represented in the same way.

**Semantic projection.** Grand et al. <d-cite key="grand2022semantic"></d-cite> showed that human feature judgments can be recovered from word embeddings by projecting them onto axes defined by antonym pairs. We adapt this framework to class-centroid axes in a transformer residual stream, projected jointly with a Gram-pseudoinverse correction (see Methods).

## Methods

As shown in Figure 1, our analysis consists of four stages, each run on all ten models: (1) extracting residual stream activations for symptom descriptions, (2) selecting the operating layer where the three symptom groups are most separable, (3) constructing Symptom Vectors at that layer to project held-out text, and (4) contrasting the depression and positive affect centroids to derive a single Depression Vector. Every stage is run identically on each model.

<figure id="fig-pipeline" style="text-align: center; margin: 20px 0;">
  <img src="{{ 'assets/img/submission/1.pipeline.png' | relative_url }}" style="max-width: 100%; height: auto;" alt="Analysis pipeline diagram in four panels">
  <figcaption style="text-align: left;"><b>Figure 1: Analysis pipeline</b>, applied to each model. <b>(a)</b> Residual stream activations are extracted at every hook point (illustrated for Gemma-3-27B-PT's 63, ℓ = 0, …, 62) while prompting with symptom descriptions from three groups. <b>(b)</b> For each layer × distance-metric combination, distance matrices are computed and tested with PERMANOVA and PERMDISP; the operating layer ℓ* is the layer maximizing pseudo-F subject to PERMDISP <i>p</i> &gt; 0.05. <b>(c)</b> Symptom Vectors are constructed from the group centroids at ℓ*, and naturalistic text samples are projected onto them to obtain symptom coefficients β. <b>(d)</b> The Depression Vector is defined as the difference between the Depression Centroid and the Positive Affect Centroid.</figcaption>
</figure>

### Data and preprocessing

#### Text corpora

The **Core Clinical** corpus was used to construct the Symptom Vectors. It comprised symptom descriptions extracted from six validated depression assessment instruments: the DSM-5 <d-cite key="apa2013dsm5"></d-cite>, ICD-10 <d-cite key="who1992icd10"></d-cite>, Hamilton Depression Rating Scale (HAM-D) <d-cite key="hamilton1960rating"></d-cite>, Montgomery-Åsberg Depression Rating Scale (MADRS) <d-cite key="montgomery1979new"></d-cite>, Patient Health Questionnaire (PHQ-9) <d-cite key="kroenke2001phq9"></d-cite>, and the patient-derived PROCEED outcome set <d-cite key="chevance2020identifying"></d-cite>. Each instrument was segmented into shorter, self-contained excerpts. Each excerpt was assigned to a single symptom group by one psychiatrist author; a second psychiatrist independently reviewed all assignments against DSM-5 item definitions. Disagreements were resolved through joint review until the two annotators reached consensus, and a final label was assigned only by agreement. This yielded 51 excerpts (19 mood, 24 somatic, 8 suicidality) distributed across DSM-5 (3), ICD-10 (3), HAM-D (16), MADRS (12), PHQ-9 (9), and PROCEED (8).

We also assembled a **Naturalistic** corpus as held-out texts for Symptom Vector evaluation. This corpus comprised first-person and clinical-narrative descriptions of the same three symptom groups, drawn from three sources: William Styron's memoir *Darkness Visible: A Memoir of Madness* <d-cite key="styron1990darkness"></d-cite>, the *Handbook of Depression*, 3rd Edition <d-cite key="gotlib2014handbook"></d-cite>, and ReDSM5, a corpus of Reddit posts annotated at the sentence level against the nine DSM-5 depression symptoms by a licensed psychologist <d-cite key="bao2025redsm5"></d-cite>. We selected *Darkness Visible* for its sustained first-person account of lived depressive experience and the *Handbook of Depression* for its third-person clinical description of symptoms; ReDSM5 was used to supplement the books. *Darkness Visible* and *Handbook* excerpts were annotated using the same two-author categorize-then-review procedure for the Core Clinical corpus. ReDSM5 samples, inheriting DSM-5 labels assigned by a licensed psychologist <d-cite key="bao2025redsm5"></d-cite>, were programmatically mapped onto the three symptom groups under the same DSM-5-item assignment. To balance the symptom groups, we retained every book-sourced passage and added randomly selected ReDSM5 posts to each group until its count matched that of the largest book-sourced group. Table 1 reports the resulting sample counts.

<p style="text-align: center; margin-bottom: 0.3em;"><b>Table 1:</b> Naturalistic corpus composition. Excerpt counts by source and symptom group.</p>

| Source | Mood | Somatic | Suicidality | Source total |
|:--|--:|--:|--:|--:|
| Darkness Visible | 33 | 11 | 13 | 57 |
| Handbook of Depression | 108 | 42 | 22 | 172 |
| ReDSM5 | — | 88 | 106 | 194 |
| **Symptom total** | **141** | **141** | **141** | **423** |

Excerpts of the Core Clinical and Naturalistic corpora were annotated with one of three symptom groups: mood/emotional/cognitive (hereafter "mood"; DSM-5 A1, A2, A7, A8: depressed mood, anhedonia, guilt or shame, concentration difficulties); somatic (DSM-5 A3–A6: appetite or weight change, sleep disturbance, psychomotor change, and fatigue in the context of depression); and suicidality (DSM-5 A9: active suicidal ideation with or without specific plan, and passive death wishes). The mood and somatic groups were chosen to reflect depression's distinct impact on the mind or body, following prior meta-analytic work on clinical instruments <d-cite key="shafer2006metaanalysis"></d-cite>. Given its clinical significance and risk profile, suicidality was assigned its own category.

The **Positive Affect** corpus comprises short excerpts of a few sentences expressing positive affect such as joy, love, wonder, aspiration, and optimism, from deliberately diverse sources spanning translated classical Chinese poetry (Li Bai, Cao Cao), literary fiction (Tolstoy, García Márquez, Rothfuss), popular-science writing (Dawkins), political oratory (Kennedy), and foundational civic prose (the U.S. Constitution). This corpus was constructed to be opposite in valence relative to the depression corpora: it consists of emotionally positive text, whereas the depression corpora are negative. It also spans a broad range of genres and styles, so that any separation observed along the symptom axes reflects shared affective valence. There are 9 total excerpts in this group. The plain text of this corpus is released with the code.

To contrast depressive text with affectively positive language, we included **HappyDB**, a corpus of crowdsourced descriptions of happy moments <d-cite key="asai2018happydb"></d-cite>. We took the first 423 non-duplicate entries to match the size of the Naturalistic corpus.

#### Text anonymization and standardization

A pilot experiment on the unedited Core Clinical corpus showed that PROCEED's residual stream is geometrically separated from all other clinical instruments regardless of clinical similarity. This separation is an artifact of its distinctive format of bullet-point symptom lists. To prevent stylistic confounds such as format, pronouns, and register from being picked up as signal, we rewrote every excerpt into a standard third-person register using Anthropic Claude Opus 4. All excerpts were rewritten with the same prompt: "Rewrite the following text into natural language as a description of a third person. Correct any grammar mistakes." All rewrites were reviewed and corrected by the authors.

These rewrites were deliberately minimal, comprising surface-level edits only: standardizing pronouns, removing severity ratings from clinical questionnaires, correcting grammar in ReDSM5 and HappyDB, and restructuring fragments into continuous prose in PROCEED. Names and identifying details were also removed. Clinical content was left unchanged.

The full set of rewritten excerpts in the Core Clinical corpus is provided in Appendix A. Table 2 shows representative examples of the rewriting from all corpora.

<p style="text-align: center; margin-bottom: 0.3em;"><b>Table 2:</b> Text standardization examples. Original and standardized excerpts from each publicly available corpus.</p>

| Dataset | Original | Standardized |
|:--|:--|:--|
| Core Clinical | INSOMNIA – Delayed (Waking in early hours of the morning and unable to fall asleep again)<br>0 = Absent<br>1 = Occasional<br>2 = Frequent | Delayed Insomnia: The person wakes up in the early hours of the morning and is unable to fall asleep again. |
| Naturalistic (Darkness Visible) | I now see, as a means to calm the anxiety and incipient dread that I had hidden away for so long somewhere in the dungeons of my spirit. | They now see this as a means to calm the anxiety and incipient dread that they had hidden away for so long somewhere in the dungeons of their spirit. |
| Naturalistic (Handbook of Depression) | Individuals with depression exhibit cognitive deficits and biased processing of emotional material. | They exhibit cognitive deficits and biased processing of emotional material. |
| Positive Affect | He spoke gently, laughed often, and never exercised his wit at the expense of others. | They speak gently, laugh often, and never exercise their wit at the expense of others. |
| HappyDB | I went on a successful date with someone I felt sympathy and connection with. | They went on a successful date with someone they felt sympathy and a connection with. |

#### Residual stream extraction

We analyzed ten open-weight models, each the official Hugging Face release: Gemma-3 at 4B, 12B, and 27B parameters, each pre-trained (PT) and instruction-tuned (IT) <d-cite key="gemmateam2025gemma3"></d-cite> (34, 48, and 62 transformer blocks; residual stream widths 2,560, 3,840, and 5,376); Mistral-Small-24B Base and Instruct (2501 release; 40 blocks, width 5,120); and OLMo-2-32B base and Instruct (0325 release; 64 blocks, width 5,120) <d-cite key="olmo2024olmo2"></d-cite>. The Gemma-3 models vary scale within one family; the five PT/IT pairs test the effect of post-training, which could alter how a model represents depression symptoms; and the Mistral and OLMo pairs test whether the results hold outside Gemma. Instruction-tuned models received the same raw text as the PT models, without a chat template.

Each excerpt was tokenized with the model's own tokenizer and processed in a single forward pass through the model, which was loaded and run with the Hugging Face Transformers library <d-cite key="wolf2020transformers"></d-cite> on a PyTorch backend <d-cite key="paszke2019pytorch"></d-cite> using Apple's Metal Performance Shaders (MPS). During each pass, forward hooks recorded the input embeddings and the residual stream output of every transformer block, yielding one activation per hook point (the number of blocks plus one). Activations corresponding to the initial `<bos>` token were discarded at every hook point prior to all subsequent analyses, to mitigate the attention-sink and massive-activation phenomena documented in transformer residual streams <d-cite key="xiao2024attentionsinks,sun2024massive"></d-cite>; OLMo-2's tokenizer does not add a beginning-of-sequence token, so we prepended its `bos_token` so that the first position could be discarded as in the other models. All activations were cast to float64 before any distance or centroid computation.

### Per-layer separability

Different residual stream layers encode different information. To identify where each model distinguishes depression symptoms, we quantified symptom-group separability at each residual stream layer on the Core Clinical corpus using PERMANOVA. We then used the model's operating layer (defined below) for the Symptom Vector extraction.

#### Pairwise distance matrices

Residual stream activations are high-dimensional (2,560–5,376 dimensions across the ten models) objects of varying length, whose token activations depend on preceding tokens through the self-attention mechanism. Tokens within an activation are therefore not independent samples; each activation has to be compared as a whole. There is no established method for measuring the distance between residual streams generated by prompts of different lengths for the purpose of comparing textual meaning. We tested the efficacy of eight distance-metric × normalization combinations: Centroid Cosine Distance, Centroid Euclidean Distance, Earth Mover's Distance, and Energy Distance, each computed on both raw and $\ell^2$-normalized residual stream activations, at every residual stream layer $\ell$ of each model.

These metrics are not directly comparable, so we reduced each to a common representation: a pairwise distance matrix. For a given metric and layer, entry $(i, j)$ is the distance between the residual stream activations of prompts $i$ and $j$ from the Core Clinical corpus. The distance matrices rule out classical ANOVA and MANOVA, which need fixed-length coordinate vectors and normally distributed underlying data, neither of which residual streams satisfy. PERMANOVA operates directly on the pairwise distances, and its pseudo-$F$ statistic is sensitive to the distance between group centroids. Because pseudo-$F$ is a ratio of between- to within-group variation, it is unaffected by the absolute scale of the input distances. PERMANOVA can thus be used to compare different metrics directly.

The centroid metrics use the mean of all token activations. Let $h_t^{\ell}$ denote the residual stream activation of token $t$ at layer $\ell$, for a sequence of $T$ tokens. The layer's centroid is defined as:

$$
\bar{h}^{\ell} = \frac{1}{T} \sum_{t=1}^{T} h_t^{\ell}.
$$

The centroid distance is then computed using PyTorch tensor operations <d-cite key="ansel2024pytorch2"></d-cite>.

Distribution-based metrics (Earth Mover's, Energy) were computed directly between full token distributions. Energy Distance was computed with the `pted` package <d-cite key="stone2026pted"></d-cite>, an implementation of the energy statistic with permutation tests <d-cite key="szekely2013energy"></d-cite>. Earth Mover's distance was computed with SciPy <d-cite key="virtanen2020scipy"></d-cite>.

#### PERMANOVA with the PERMDISP eligibility gate

Using the pairwise distance matrices, we assessed the separability of residual stream activations across symptom groups with PERMANOVA, gated by PERMDISP. At each layer and for each metric, we computed a PERMANOVA pseudo-$F$ statistic <d-cite key="anderson2001permanova"></d-cite>:

$$
\text{pseudo-}F = \frac{\sum (\text{Between-Group Distance})^2 / (g - 1)}{\sum (\text{Within-Group Distance})^2 / (n - g)},
$$

where $g = 3$ is the number of symptom groups and $n$ is the total number of excerpts.

PERMANOVA is sensitive not only to differences in group centroid location but also to differences in within-group multivariate dispersion: a group whose excerpts are widely spread can produce a significant pseudo-$F$ even when its centroid coincides with the others. A pseudo-$F$ is therefore interpretable as centroid separation only at layers where within-group dispersion is homogeneous across groups. To identify those layers, we computed the Permutational Analysis of Multivariate Dispersions (PERMDISP) <d-cite key="anderson2006permdisp"></d-cite>, which tests the null hypothesis of equal within-group dispersion, at every layer. We define the *eligible region* of a metric as the set of layers at which PERMDISP is non-significant ($p > 0.05$), meaning there is no significant difference in group dispersion; at these layers a PERMANOVA pseudo-$F$ reflects genuine displacement of group centroids. All distance statistics were computed with the scikit-bio package <d-cite key="aton2026scikitbio"></d-cite> (PERMDISP with `test="centroid"`), each on 9,999 permutations with a fixed seed.

#### Operating layer selection

For each metric, the *gated peak* (the operating peak in Figure 3) is the layer of highest PERMANOVA pseudo-$F$ within the eligible region. At each gated peak we report the PERMANOVA pseudo-$F$ and $p$-value and the PERMDISP $F$ and $p$-value; a PERMDISP $p$ just above 0.05 marks weakly supported eligibility. Because models differ in depth, each layer is also reported as relative depth (layer divided by the number of transformer blocks).

A model's *operating layer* $\ell^\ast$ is the layer at which the largest number of metrics' gated peaks coincide, with ties broken in favor of the layer at which more metrics pass the PERMDISP gate. Each model's Symptom Vectors, symptom projection, and Depression Vector are built at its own operating layer, from the same corpora and with the same sampling and procedures described below. For the five PT/IT pairs, we compare the location and strength of separation and the held-out Depression Vector AUC between the pre-trained and the instruction-tuned model.

### Semantic projection for symptom identification

The procedure below adapts the semantic projection framework of Grand et al. <d-cite key="grand2022semantic"></d-cite>, who showed that context-dependent human judgments about object features can be recovered from word embeddings by projecting word vectors onto interpretable axes: the line connecting "small" to "big," for instance, yields an axis along which a word's projection reflects perceived size. We extend this framework in three ways. First, symptom axes are constructed from class-centroid activations rather than antonymous word pairs, because clinical symptom categories (mood, somatic, suicidality) lack the natural antonymy of adjective pairs. Second, the three symptom axes are projected jointly rather than independently, with the resulting values decorrelated via a Gram-pseudoinverse correction. Third, the representations being projected are residual stream activations of a contemporary transformer language model, taken at the operating layer and in the raw activation geometry identified in the previous experiment, whereas the original framework used static distributional word vectors.

The output of the procedure is, for each input passage $x$, a three-dimensional vector of decorrelated projection coefficients,

$$
\beta(x) = (\beta_{\text{mood}}, \beta_{\text{somatic}}, \beta_{\text{suicidality}}) \in \mathbb{R}^3,
$$

each coefficient quantifying the unique contribution of one symptom axis to the passage's representation in residual stream space at the operating layer. We refer to the subspace of residual stream space spanned by the three symptom basis vectors as the *symptom subspace*. The construction of the basis, the projection, and the properties of $\beta$ are described in turn below.

**Symptom Vector construction.** At the operating layer $\ell^\ast$ identified by the separability analysis, we constructed three Symptom Vectors $v_s \in \mathbb{R}^{d}$, where $d$ is the model's residual stream width, for $s \in \lbrace \text{mood}, \text{somatic}, \text{suicidality} \rbrace$ from the Core Clinical corpus. Each $v_s$ is the centroid of all residual stream activations associated with symptom $s$, computed in the model's native activation geometry. As throughout, the initial `<bos>` token is removed before averaging, to prevent attention-sink effects <d-cite key="xiao2024attentionsinks,sun2024massive"></d-cite>, and all centroid arithmetic is performed in float64 on CPU. The three centroids were stacked to form the basis matrix:

$$
V = [\, v_{\text{mood}};\ v_{\text{somatic}};\ v_{\text{suicidality}} \,] \in \mathbb{R}^{3 \times d}.
$$

**Gram matrix construction.** Because the axes are nearly collinear, a naive projection conflates their contributions; we therefore decorrelate the projection using the inverse of the Gram matrix. The geometric relationships between $v_{\text{mood}}, v_{\text{somatic}}, v_{\text{suicidality}}$ are captured by the Gram matrix, formed from the basis matrix $V$ constructed in the previous step:

$$
G = V V^\top \in \mathbb{R}^{3 \times 3}.
$$

Its entries are the pairwise inner products $G_{ij} = v_i \cdot v_j$, so the diagonal $G_{ii} = \lVert v_i \rVert^2$ holds the squared norms of the Symptom Vectors and the off-diagonal entries hold their overlaps. An orthonormal basis would give $G = I$. As the basis approaches parallel, the smallest eigenvalue of $G$ approaches zero and the condition number $\kappa(G) = \lambda_{\max} / \lambda_{\min}$ becomes large, and inverting $G$ amplifies small perturbations in the embeddings into large errors in the symptom loadings. This near-singular condition is addressed by the pseudoinverse below.

**Pseudoinverse of the Gram matrix.** Using $G^{-1}$ is not practical here. The clinical symptom categories of depression are textually comorbid: a passage describing one symptom often touches on others, so the class-centroid basis is expected to be near-collinear. $G$ is then ill-conditioned, and $G^{-1}$ amplifies small perturbations in the projected coordinates by a factor of up to $\kappa(G)$. If the basis vectors were exactly parallel, $G$ would be singular and $G^{-1}$ would not exist at all.

We therefore use the Moore–Penrose pseudoinverse $G^{+}$ in place of $G^{-1}$. For invertible $G$, $G^{+} = G^{-1}$; for singular or near-singular $G$, $G^{+}$ yields the unique $\beta$ minimizing both the residual $\lVert V c(x) - G \beta \rVert^2$ and the coefficient norm $\lVert \beta \rVert^2$. The condition number of $G$ in each model is reported in the Results.

**Symptom projection.** Finally, we reduce each excerpt from a residual stream of $d$ dimensions × excerpt length to 3 symptom coefficients. The decorrelated projection coefficients for a passage $x$ are:

$$
\beta(x) = G^{+} V \, c(x),
$$

where $c(x)$ is the passage's centroid representation: we extracted the operating-layer residual stream activations for $x$ and computed the float64 centroid of the remaining tokens, using the identical procedure as for the symptom basis vectors.

This formulation is the multi-axis analogue of the projection step in Grand et al. <d-cite key="grand2022semantic"></d-cite>: when the basis is orthonormal, $G^{+} = I$ and $\beta$ reduces to an inner-product projection onto each axis independently. When the basis is collinear, as is generally the case for class-centroid bases over correlated taxonomies, the pseudoinverse generalizes that projection, recovering each axis's unique contribution after accounting for the others.

The resulting projection coefficients $\beta$ have three interpretable properties:

- **Per-symptom isolation.** Each $\beta_s$ reflects the unique contribution of $v_s$ alone, with the shared signal among correlated axes apportioned by the Gram correction.
- **Sign.** The sign of $\beta_s$ indicates whether the passage points along $v_s$ (positive) or against it (negative).
- **Cross-passage comparability.** Because $\beta$ is derived from a fixed basis $V$ and a uniform projection procedure, projection coefficients from different excerpts and corpora live in the same coordinate system, enabling direct cross-corpus comparison, e.g., comparing $\beta_{\text{suicidality}}$ between clinical instrument and naturalistic depression descriptions.

We projected each passage's centroid from the Core Clinical and Naturalistic corpora onto $V$ using this procedure. Core Clinical projections constitute an in-sample consistency check; only the Naturalistic projections evaluate whether the methodology generalizes to held-out text as external validation. Each passage thus received three decorrelated projection coefficients $(\beta_{\text{mood}}, \beta_{\text{somatic}}, \beta_{\text{suicidality}})$, interpreted in the Results.

### Depressive and positive affect distinction

Because projecting non-depressive text onto Symptom Vectors is not meaningful, we also tested whether an LLM can distinguish between depressive and non-depressive text. We performed a control experiment at the operating layer. Two centroids of opposite emotional valence were built from the operating layer:

1. The **Depression** centroid is the centroid of the three Symptom Vectors from the previous experiment, constructed from the Core Clinical corpus. This gives equal weight to all three symptoms:

$$
\bar{h}_{\text{depressive}} = \frac{v_{\text{mood}} + v_{\text{somatic}} + v_{\text{suicidality}}}{3}.
$$

{:start="2"}
2. The **Positive Affect** centroid is the token-weighted mean over all tokens of the entire Positive Affect corpus, excluding `<bos>`:

$$
\bar{h}_{\text{positive}} = \frac{1}{N_P} \sum_{x \in P} \sum_{t=1}^{T_x} h_{x,t}, \qquad N_P = \sum_{x \in P} T_x,
$$

where $P$ is the set of positive-affect texts, $T_x$ is the number of tokens in text $x$, $t$ indexes the tokens within a text, and $N_P$ is the total token count across the set.

We then define the **Depression Vector** to be:

$$
v_{\text{depression}} = \bar{h}_{\text{depressive}} - \bar{h}_{\text{positive}}.
$$

The Naturalistic corpus and the HappyDB corpus are reserved as held-out tests that aren't included in the construction of these axes.

An excerpt $x$ is scored by the cosine similarity between its token-weighted centroid $\bar{h}_x$ (computed as above, excluding `<bos>`) and the Depression Vector; we call this the **Depression Score** of excerpt $x$:

$$
\text{Score}(x) = \cos(\bar{h}_x, v_{\text{depression}}) = \frac{\bar{h}_x \cdot v_{\text{depression}}}{\lVert \bar{h}_x \rVert \, \lVert v_{\text{depression}} \rVert}.
$$

We compared Depression Scores between the held-out Naturalistic and HappyDB corpora (423 excerpts each) using a one-sided Mann–Whitney $U$ test. We report the test statistic in its normalized form $U / (n_1 n_2)$, where $n_1$ and $n_2$ are the number of excerpts in the two corpora being compared, which equals the area under the receiver operating characteristic curve (AUC) and is interpretable as a probability of superiority: the chance that a randomly drawn depressive excerpt scores higher than a randomly drawn non-depressive one. A value of 0.5 indicates no separation; 1.0 indicates complete separation. We computed the same statistic for Core Clinical versus Positive Affect as an in-sample consistency check.

## Results

Every result below is reported for all ten models. The model-specific figures and tables (Figures 3–5, Tables 3–4) show one model at a time: the floating Model selector at the upper right, which appears while one of them is on screen, switches all of them at once, and Table 5 summarizes every model.

<div id="model-picker" style="display: none; position: fixed; right: 16px; top: 16px; z-index: 1000; background: white; border: 1px solid #ccc; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.15); padding: 6px 10px; font: 13px Arial, sans-serif; color: black;"><label>Model <select id="model-select" style="font-size: 1em; padding: 2px 4px; max-width: 60vw;"></select></label></div>

<script>
// One floating model selector for every figure and table on the page. Each figure iframe
// (figure_style.write_tmlr_dropdown_html) posts {type: 'tmlr-model-ready', labels} on load: the
// first fills the selector, each is told the current model, and the selector shows (top right,
// under the TMLR navbar) while any of them is on screen. Changing it sets every figure.
(function () {
  var picker = document.getElementById('model-picker'), select = document.getElementById('model-select');
  document.body.appendChild(picker);  // fixed to the viewport, clear of any transformed ancestor
  var frames = [];
  function place() {
    var nav = document.getElementById('navbar');  // fixed-top or sticky-top in the TMLR layout
    picker.style.top = (nav ? Math.max(0, nav.getBoundingClientRect().bottom) : 0) + 12 + 'px';
    picker.style.display = frames.some(function (f) {
      var r = f.getBoundingClientRect();
      return r.bottom > 0 && r.top < window.innerHeight;
    }) ? 'block' : 'none';
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
    document.querySelectorAll('iframe').forEach(function (f) { tell(f.contentWindow); });
  });
})();
</script>

### Symptom groups are separable in all ten models

We first asked whether depression-symptom categories are geometrically separable in language-model residual streams and, if so, at which layer separability is strongest. For every model, we computed the PERMANOVA pseudo-$F$ and its PERMDISP $p$-value at every hook point for the eight distance-metric × normalization combinations, and took each metric's gated peak, its maximum pseudo-$F$ within the eligible region (Figure 3; Table 3). In all ten models, every metric had an eligible layer, and every gated peak was significant (PERMANOVA $p \le 0.031$).

<figure id="fig-separation" style="text-align: center; margin: 20px 0;">
  <iframe src="{{ 'assets/html/submission/82.multimodel_separation.html' | relative_url }}" width="100%" height="808" style="border: none;" title="Per-layer PERMANOVA pseudo-F for eight distance metrics"></iframe>
  <figcaption style="text-align: left;"><b>Figure 3: Symptom separability across layers, with each metric's gated peak.</b> PERMANOVA pseudo-F (y-axis) across all hook points (x-axis) for each of the eight distance-metric × normalization combinations, ordered by operating pseudo-F. Shaded bands mark layers where PERMDISP is significant (<i>p</i> &lt; 0.05) and pseudo-F is therefore ineligible as a measure of centroid separation. The operating peak (green) is the maximum pseudo-F within each metric's eligible region (<i>p</i> &gt; 0.05); the excluded peak (gray) is a higher but ineligible global maximum, where one exists. Hover a layer for its exact statistics. Choose the model with the floating Model selector at the upper right; each model is shown at its own operating layer (Table 5).</figcaption>
</figure>

<figure id="tab-permanova" style="text-align: center; margin: 20px 0;">
  <p style="margin-bottom: 0.3em;"><b>Table 3:</b> PERMANOVA and PERMDISP statistics at each metric's eligible maximum. For each distance-metric × normalization combination, the residual stream layer where PERMANOVA pseudo-F is maximized within the eligible region (PERMDISP <i>p</i> &gt; 0.05), with each statistic's <i>p</i>-value beneath it. Choose the model with the floating Model selector at the upper right; each model is shown at its own operating layer (Table 5).</p>
  <iframe src="{{ 'assets/html/submission/82.multimodel_table_permanova.html' | relative_url }}" width="100%" height="508" style="border: none;" title="Table 3: PERMANOVA and PERMDISP at each metric's operating peak"></iframe>
</figure>

The gated peaks clustered at an early operating layer in most models: at 20–34% of depth in eight of the ten (Mistral-Small-24B, layer 8 of 40 for both PT and IT; OLMo-2-32B, 16 of 64 for both; Gemma-3-12B, 13 of 48 for PT and 14 of 48 for IT; Gemma-3-27B, 21 of 62 for both). The Gemma-3-4B pair was the exception, at layer 20 of 34 (59%) for both PT and IT. Between two and five of the eight metrics' gated peaks coincided at the operating layer (five for Gemma-3-27B-IT and both Mistral models; four for Gemma-3-4B-IT; three for Gemma-3-27B-PT, Gemma-3-12B-PT, Gemma-3-4B-PT, and both OLMo-2 models; two for Gemma-3-12B-IT), and the highest gated pseudo-$F$ there ranged from 5.41 to 8.45 (Table 5).

The centroid metrics defined the operating layer most often: Centroid Cosine (raw) peaked there in seven models, and Centroid Cosine (unit-normalized) and both Centroid Euclidean variants in six each, against four or fewer models for each distribution metric (Earth Mover's, Energy). Gated peaks away from the operating layer mostly lay deeper in the network (34 of 44) and were more weakly eligible: their PERMDISP $p$ fell below 0.10 in 52% of cases, against 31% at the operating layer, so part of the separation there reflects unequal group dispersion rather than centroid displacement. The operating layer is therefore each model's most consistent point of centroid separation, and every subsequent analysis is run there.

### Semantic projection coefficients align with clinician ratings

We next asked whether projecting text onto Symptom Vectors at the operating layer yields per-symptom coefficients that agree with independent clinician annotation, and whether that agreement holds on a held-out set of naturalistic narratives. In each model, we projected every Core Clinical and Naturalistic passage onto the model's three Symptom Vectors using the decorrelation procedure, yielding coefficients $\beta = (\beta_{\text{mood}}, \beta_{\text{somatic}}, \beta_{\text{suicidality}})$ per passage. As anticipated for textually comorbid symptom categories, the class-centroid basis was collinear in every model, and strongly so in Gemma-3: the Gram matrix $G$ had a condition number of 8,959–18,144 in the six Gemma-3 models, against 33.5–37.1 in the Mistral and OLMo models, so a direct inverse would be numerically unstable for Gemma-3 in particular, motivating the Moore–Penrose pseudoinverse correction. Table 4 reports the median together with Q1 and Q3 coefficients for each clinician-annotation group within each dataset; Figure 4 shows the full per-axis distributions.

<figure id="tab-coefficients" style="text-align: center; margin: 20px 0;">
  <p style="margin-bottom: 0.3em;"><b>Table 4:</b> Median projection coefficients by annotation group and symptom axis, with Q1 and Q3 beneath. In-sample is the Core Clinical corpus; held-out is the Naturalistic corpus. Within each dataset, the annotation group matching a given symptom axis attains the highest median coefficient. This ordering holds from the in-sample Core Clinical corpus to the held-out Naturalistic corpus. Choose the model with the floating Model selector at the upper right; each model is shown at its own operating layer (Table 5).</p>
  <iframe src="{{ 'assets/html/submission/82.multimodel_table_coefficients.html' | relative_url }}" width="100%" height="428" style="border: none;" title="Table 4: median projection coefficients"></iframe>
</figure>

The projection recovered the clinician annotations in every model. On the in-sample Core Clinical corpus, from which the Symptom Vectors were built, the excerpts carrying the matching annotation attained the highest median coefficient on each of the three axes in all ten models, an internal consistency check that Symptom Vectors recover the categories from which they were constructed (Figure 4(a)–(c)).

The same rank ordering was preserved on the held-out Naturalistic corpus, the true test of generalization: in all ten models, the matching annotation group again ranked first on every axis (Figure 4(d)). In both corpora and every model, each matching group exceeded each of the other two groups on its own axis (Dunn's test, Benjamini–Hochberg-adjusted $p < 0.05$ for all 120 matched-versus-other comparisons: six per corpus per model). On the held-out corpus, the matching group's median led the next-highest group's by 0.46–0.73 on the mood axis, 0.31–0.43 on the somatic axis, and 0.33–0.54 on the suicidality axis across models. Projecting held-out naturalistic narratives onto Symptom Vectors constructed from clinical instruments yields coefficients that agree with independent clinician annotation, in every model.

<figure id="fig-coefficients" style="text-align: center; margin: 20px 0;">
  <iframe src="{{ 'assets/html/submission/82.multimodel_coefficient_abc.html' | relative_url }}" width="100%" height="648" style="border: none;" title="Figure 4(a)–(c): per-axis coefficient distributions"></iframe>
  <iframe src="{{ 'assets/html/submission/82.multimodel_coefficient_rank.html' | relative_url }}" width="100%" height="408" style="border: none;" title="Figure 4(d): rank matrices of median coefficients"></iframe>
  <figcaption style="text-align: left;"><b>Figure 4: Semantic projection yields clinician-aligned symptom coefficients.</b> Symptom coefficients (β<sub>mood</sub>, β<sub>somatic</sub>, β<sub>suicidality</sub>) projected onto Symptom Vectors, grouped by clinician annotation and decorrelated. <b>(a)–(c)</b> Per-axis coefficient distributions (box plots with overlaid points) for the mood (a), somatic (b), and suicidality (c) axes, with the in-sample Core Clinical corpus on the top row and the held-out Naturalistic corpus on the bottom; boxes show the median and interquartile range, points show individual excerpts (hover to read each one), and n is the number of excerpts per group. Brackets denote pairwise comparisons between annotation groups (***, <i>p</i> &lt; 0.001; **, <i>p</i> &lt; 0.01; *, <i>p</i> &lt; 0.05; ns, not significant). <i>p</i>-values are from Dunn's post-hoc test on pooled within-axis ranks, Benjamini–Hochberg-corrected jointly across all nine comparisons. <b>(d)</b> Median β for each annotation group on each axis, with cells shaded and ranked by within-column standing (1st = highest median β); the bordered diagonal marks each group scoring highest on its own axis. The preserved within-corpus rank ordering is the evidence of generalization. Choose the model with the floating Model selector at the upper right; each model is shown at its own operating layer (Table 5).</figcaption>
</figure>

The joint structure of these coefficients is visible in the three-dimensional projection (Figure 5): in each model, clinical and naturalistic excerpts sharing an annotation occupy a common region of the symptom subspace.

<figure id="fig-subspace" style="text-align: center; margin: 20px 0;">
  <iframe src="{{ 'assets/html/submission/82.multimodel_3d.html' | relative_url }}" width="100%" height="548" style="border: none;" title="Figure 5: passages in the three-dimensional symptom subspace"></iframe>
  <figcaption style="text-align: left;"><b>Figure 5: Passages sharing a clinician annotation occupy the same region of the symptom subspace.</b> Distribution of the three decorrelated projection coefficients (β<sub>mood</sub>, β<sub>somatic</sub>, β<sub>suicidality</sub>) for all passages, plotted in the symptom subspace at the model's operating layer. Each point is one passage, colored by clinician annotation (mood, somatic, suicidality) and shaped by corpus (squares, in-sample Core Clinical; circles, held-out Naturalistic). Drag to rotate; hover a point to read the passage. Choose the model with the floating Model selector at the upper right; each model is shown at its own operating layer (Table 5).</figcaption>
</figure>

### A single Depression Vector distinguishes depressive and non-depressive text

Having established that symptom categories are separable and projectable at the operating layer (Figures 4 and 5), we finally asked whether the same representational space supports the distinction between depressive and non-depressive text. In each model, we scored every excerpt by the cosine similarity between its token-weighted centroid and the model's Depression Vector (Figure 6).

Core Clinical excerpts ($n = 51$) and Positive Affect excerpts ($n = 9$) were completely separated by the Depression Score in every model (AUC = 1.000, one-sided Mann–Whitney $U$, $p = 1.06 \times 10^{-6}$). Since both corpora participated in constructing the axis, this result serves only as an in-sample consistency check, not as evidence of generalization.

Neither the held-out Naturalistic nor the HappyDB corpus contributed to the construction of the Depression Vector, so the difference between them tests generalization of the valence axis to unseen text. In every model, Naturalistic depressive excerpts ($n = 423$) received higher Depression Scores than HappyDB happy-moment descriptions ($n = 423$), with AUCs from 0.730 to 0.965 (one-sided Mann–Whitney $U$, all $p < 10^{-30}$). Eight models reached 0.943–0.965, so that a randomly drawn depressive narrative outscores a randomly drawn happy-moment description about 19 times in 20; the Gemma-3-27B pair was lowest (0.789 PT, 0.730 IT).

The HappyDB entries that score highest are consistent across models: in every model the top-scoring entry describes a bodily or health event, "Their sore throat cleared up" in five models and "They are able to sleep longer than usual" in the other five. Positive in valence, these entries share the somatic vocabulary of the depressive corpora, which the Depression Vector partly carries.

<figure id="fig-depvector" style="text-align: center; margin: 20px 0;">
  <iframe src="{{ 'assets/html/submission/82.multimodel_auc.html' | relative_url }}" width="100%" height="1000" style="border: none;" title="Figure 6: Depression Score by corpus"></iframe>
  <figcaption style="text-align: left;"><b>Figure 6: A single operating-layer Depression Vector separates depressive from non-depressive text.</b> Cosine similarity between each excerpt's token-weighted centroid and the Depression Vector, by corpus. Core Clinical and Positive Affect are in-sample (both were used to construct the Depression Vector); Naturalistic and HappyDB are held out. Boxes show the median and interquartile range; points show individual excerpts (hover to read each one). Brackets report AUC for the in-sample and held-out contrasts; ***, <i>p</i> &lt; 0.001, one-sided Mann–Whitney U test. One panel per model, each at its own operating layer (Table 5); y-axes differ between panels because the score scale depends on the model family.</figcaption>
</figure>

Two features of the score distributions are worth noting. First, the four corpora ordered as expected along the axis in nine of the ten models, with the median Depression Score decreasing from Core Clinical to Naturalistic to HappyDB to Positive Affect, consistent with the Depression Vector capturing a graded valence signal rather than a corpus-identity artifact; in Gemma-3-27B-IT, Naturalistic scored marginally above Core Clinical (median 0.530 vs. 0.528). Second, the score scale depends on the family. In the six Gemma-3 models, all held-out scores fall within a narrow band at most 0.06 wide, so the held-out distributions overlap substantially; Mistral and OLMo spread them over 0.51–0.56. The Depression Score therefore separates depressive from non-depressive text at the population level in every model but is not, on its own, a reliable per-excerpt classifier; its role here is to establish that a valence gate is recoverable from the same operating-layer geometry used for symptom projection.

### Summary across models

Table 5 collects the per-model results: the operating layer and how many metrics agree on it, the strength of separation there, clinician agreement on the held-out corpus, and the held-out Depression Vector AUC. Every model separates the symptom groups, agrees with clinician annotation on all three axes of held-out text, and supports a Depression Vector well above chance.

<figure id="tab-models" style="text-align: center; margin: 20px 0;">
  <p style="margin-bottom: 0.3em;"><b>Table 5:</b> Separability, symptom projection, and the Depression Vector in every model. Operating layer: the layer at which the most metrics' gated peaks coincide, with the number of transformer blocks and the relative depth; agreeing metrics: how many of the eight metrics' gated peaks sit at that layer; max gated pseudo-F: the highest PERMANOVA pseudo-F at that layer among metrics passing the PERMDISP gate. Clinician agreement: annotation groups in the held-out Naturalistic corpus whose median coefficient ranks first on their own symptom axis. Contrastive AUC: held-out Naturalistic versus HappyDB on the Depression Score.</p>
  <iframe src="{{ 'assets/html/submission/82.multimodel_comparison_table.html' | relative_url }}" width="100%" height="410" style="border: none;" title="Table 5: separability, projection, and Depression Vector in every model"></iframe>
</figure>

### Pre-trained versus instruction-tuned models

The five PT/IT pairs isolate the effect of post-training on each of these results.

Post-training left the location of separation largely unchanged: within each pair the operating layer was identical (Gemma-3-27B, Gemma-3-4B, Mistral-Small-24B, OLMo-2-32B) or adjacent (Gemma-3-12B, layers 13 and 14). It did, however, raise the strength of separation: in all five pairs the highest gated pseudo-$F$ at the operating layer was higher in the instruction-tuned model (Gemma-3-27B, 8.40 vs. 6.91; Gemma-3-12B, 8.03 vs. 5.91; Gemma-3-4B, 8.45 vs. 6.55; Mistral-Small-24B, 6.16 vs. 5.84; OLMo-2-32B, 5.46 vs. 5.41).

Stronger separation did not translate into better downstream generalization. Every model, PT or IT, preserved the clinician rank ordering, and post-training barely changed the held-out AUC in four pairs (Gemma-3-12B, 0.959 PT vs. 0.965 IT; Gemma-3-4B, 0.945 vs. 0.953; Mistral-Small-24B, 0.943 vs. 0.946; OLMo-2-32B, 0.945 vs. 0.946), while lowering it in the fifth (Gemma-3-27B, 0.789 vs. 0.730). Neither post-training state is consistently better downstream: post-training sharpens the separation of the symptom groups without improving how well the resulting axes generalize.

## Discussion

One challenge in treating depression is its clinical heterogeneity: patients sharing a diagnosis exhibit widely divergent symptom profiles. Yet, standard questionnaires often collapse this complexity into a single severity score <d-cite key="fried2015depression,chevance2020identifying"></d-cite>. Here, we show that when large language models process patient narratives, distinct depression symptoms are recoverable directly from their residual streams, in all ten open-weight models we tested across three families. Three key findings support this conclusion. First, the models' internal representations of different symptom categories (mood/emotional/cognitive, somatic, and suicidality) are geometrically separable. This separability concentrates at an early operating layer (20–34% of depth in eight of the ten models), where the metrics' gated peaks coincide and separation reflects distinct group centroids rather than unequal within-group dispersion. Second, projecting new patient narratives onto Symptom Vectors at the operating layer with a Gram-pseudoinverse correction yields decorrelated coefficients that preserve clinician-aligned rank ordering on held-out naturalistic narratives, in every model. Finally, because these Symptom Vectors are built entirely from residual stream activations prompted by depressive text, their projection coefficients are only meaningful for depressive speech. To address this caveat, we introduce a "Depression Vector," derived from the same operating-layer activations, that acts as a preliminary valence gate to first identify whether the input text is depressive; it separates held-out depressive from non-depressive text in every model (AUC 0.730–0.965). Ultimately, this two-stage pipeline extracts decorrelated, clinician-aligned symptom profiles of patient speech directly from LLM activations.

Because every step is computed from residual stream geometry alone, without fine-tuning or labeled training data, the same pipeline can be run on any open-weight model to check, before the model is adopted for clinical NLP research, whether it represents clinician-defined symptom structure and depressive valence in a readable form (Table 5). Whether such a screen predicts downstream clinical performance remains to be tested.

These findings extend work showing that semantic categories, including sentiment, are encoded as approximately linear directions in LLM residual streams <d-cite key="park2025geometry,hollinsworth2024sentiment,dipalma2025llamas"></d-cite>. While prior work has characterized coarse valence, we map an instrument-derived symptom taxonomy (DSM-5, ICD-10, HAM-D, MADRS, PHQ-9, and PROCEED) onto a model's internal representations and recover symptom signals along three axes, from the internal activations of unmodified models rather than from generated output or a fine-tuned model.

The Gram-pseudoinverse decorrelation of the three strongly collinear symptom axes (Gram matrix condition number $0.9$–$1.8 \times 10^4$ in the Gemma-3 models, against about 35 in the Mistral and OLMo models) turns overlapping clinical categories into a stable set of measurement axes, so that text excerpts from different sources can be scored and compared on the same scale. In the Gemma-3 models, whose bases are the most collinear, a passage's three coefficients sum to nearly 1 (mean 0.99–1.03), which is why their passages lie close to a plane in Figure 5. Gemma-3 coefficients are therefore best read as proportions: they show which symptom dominates a passage, not how intensely the passage expresses depressive content overall. The Depression Vector, by contrast, follows the semantic-projection framework of Grand et al. <d-cite key="grand2022semantic"></d-cite> closely: because depressive and positive affect are opposed in emotional valence, a contrastive axis between them is well defined, and a simple cosine projection onto it generalizes to unseen text.

## Limitations

Several limitations must be addressed prior to clinical application. First, we collapsed the nine DSM-5 symptoms into three groups for statistical power, which limits the symptom specificity of our analysis. Second, although every analysis was repeated in ten models, they span only three families and 4–32 billion parameters, all evaluated on the same English corpora; whether the results extend to other architectures, larger scales, and other languages is untested. Third, preprocessing is a potential confound: every passage was rewritten into third-person prose by another large language model. Although the edits were minimal, reviewed by psychiatrist authors, and designed to preserve clinical content, this step may introduce bias. Fourth, and most importantly, agreement with clinician annotation is not equivalent to clinical validity: nothing reported here has been validated against diagnostic outcomes, treatment response, or longitudinal trajectories, and the method should not be construed as a clinical instrument.

## Conclusion

Future research should address these limitations by granularizing Symptom Vectors using larger clinical corpora, extending the analysis to further model families, scales, and languages, and calibrating projections into an absolute and longitudinally stable severity scale suitable for patient monitoring. With these steps, the clinician-aligned Symptom Vectors identified here provide an interpretable foundation for symptom analysis of patient language, and a generalizable template for extracting other clinical concepts from the internal representations of large language models. The pipeline is model-agnostic, offering an interpretability screen for selecting language models in clinical NLP before deployment-oriented validation.

## Broader Impact Statement

This study involved the secondary analysis of publicly available, de-identified datasets. Data sources included HappyDB, a corpus of crowdsourced text responses, and ReDSM5, a dataset of public social media posts. Because the research relied exclusively on pre-existing, de-identified, and publicly accessible data, the study was deemed exempt from Institutional Review Board (IRB) review.

Inferring depression symptoms from language carries risks beyond those of the analysis itself. The prospect of an activation-level risk detector raises questions of consent, false-positive burden, and surveillance that any deployment would need to resolve through appropriate governance. The Symptom Vectors and Depression Vector reported here have not been validated against clinical outcomes and should not be used to screen, diagnose, or monitor individuals.

## Reproducibility Statement

All analyses were performed in Python, and code to reproduce every figure and table is provided in an anonymized repository for review: [ANONYMIZED REPOSITORY URL]. All models are public Hugging Face releases, and all permutation tests use 9,999 permutations with a fixed seed. The third-person rewritten Core Clinical corpus (excluding copyright-protected source text) and the Positive Affect corpus (public-domain sources) are released with the code, together with the Gram-corrected projection pipeline and, for each of the ten models, the Symptom Vectors, their Gram pseudoinverse, and the Depression Vector at the model's operating layer, so that new text can be scored without re-extracting the corpora. The following corpora are withheld: raw clinical instrument text is not redistributed in accordance with source copyright, and the ReDSM5 corpus is not redistributed under its license terms; *Darkness Visible* and the *Handbook of Depression* excerpts are also copyright-protected and not redistributable. Citations are provided in the Methods to allow independent assembly of an equivalent corpus.

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

