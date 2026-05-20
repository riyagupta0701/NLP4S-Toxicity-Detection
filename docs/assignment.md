# DSAIT4100 NLP for Society — Project Proposal

**Group 3:** Atharva Dagaonkar, Ahmed Driouech, Alexandru-Valentin Florea, Riya Gupta

## Domain: Implicit vs Explicit Hate Speech Detection in a Multilingual Setting

Hate speech detection is a sensitive topic; false negatives leave targeted communities unprotected, while false positives suppress legitimate speech. Research in this domain distinguishes **explicit hate**, which uses slurs, dehumanising, or negative attributes, from **implicit hate**, which conveys hostility through stereotyping or coded language without lexical cues a keyword system would catch [1]. Implicit hate is especially hard to predict. El Sherief et al. [1] report substantial drops in performance on implicit instances, and recent work shows that even instruction-tuned Large Language Models (LLMs) often over-predict hate on neutral content with identity terms while under-predicting hostile statements [6].

However, most of these studies only base their research on English datasets. Recent work has evaluated LLMs on functional benchmarks such as Multilingual HateCheck (MHC) [3] across multiple languages [5, 2], but these studies do not make a distinction between implicit and explicit hate speech in their experiments. The MHC dataset does allow for distinction between implicit and explicit hate speech instances, but no study in this domain seems to address this gap. Therefore, this project addresses that gap with an implicit vs explicit decomposition of LLM behaviour across all MHC languages, with English as baseline.

## Dataset

We use the **Multilingual HateCheck (MHC)** dataset [3], a diagnostic suite of functional tests for hate speech detection models covering 10 languages (Arabic, Dutch, French, German, Hindi, Italian, Mandarin, Polish, Portuguese, Spanish) in addition to the original English HateCheck [4]. MHC provides fine-grained *functionality* labels that allow us to operationalise the implicit–explicit distinction. We group the following functionalities into two categories:

- **Explicit hate:** `derog_neg_attrib_h` (derogatory negative attribute), `derog_dehum_h` (dehumanization), `slur_h` (slur usage), `profanity_h` (hateful profanity).
- **Implicit hate:** `derog_impl_h` (implicit derogation) [1].
- **Non-hateful control:** `profanity_nh` (profanity in a non-hateful context).

This grouping allows controlled comparison of model performance on implicit vs. explicit categories across all 11 languages.

### Dataset language coverage and the training-data gap

MHC is an **evaluation/diagnostic** benchmark only; it provides no training split. Training data must come from elsewhere, and here the available corpora diverge sharply from the MHC evaluation languages:

- **MHC (evaluation), 11 languages:** Arabic, Dutch, French, German, Hindi, Italian, Mandarin, Polish, Portuguese, Spanish, English [3, 4].
- **HASOC (candidate training data):** the early editions (2019–2020) cover only **English, German, and (code-mixed) Hindi**; later editions add Marathi, Bengali, Assamese, Bodo, Sinhala, and Gujarati — **none of which appear in MHC**. HASOC is also a *coarse binary* HOF/NOT task with **no implicit/explicit distinction**, ~8k posts per language.
- **ToxiGen:** large-scale (~274k) machine-generated dataset rich in **implicit** hate across 13 target groups, but **English-only** [7].

The consequence: fine-tuning on HASOC only directly covers **3 of the 11** MHC languages (English, German, Hindi) and gives the encoder almost no *implicit* hate supervision. This motivates two complementary directions:

1. **Cross-lingual transfer** — fine-tune a multilingual encoder (XLM-RoBERTa) on the available languages and rely on its shared representations to transfer to the remaining MHC languages.
2. **Synthetic data generation** — use a massively multilingual LLM to generate ToxiGen-style **implicit** hate / non-hate pairs directly in the under-covered MHC languages. **Aya (Cohere)** is a strong fit here: **Aya-23** covers all 11 MHC languages and **Aya-101** covers 101 languages [8, 9], so it (alongside Llama-3) can produce implicit, demonstration-conditioned examples in languages where no labelled implicit data exists. This makes "implicit multilingual" generation feasible where ToxiGen (English-only) cannot reach.

## Research Questions

### RQ1: How does hate speech detection performance differ between implicit and explicit hate across languages?

- **SQ1.1:** What is the performance gap (F1, precision, recall) between implicit (`derog_impl_h`) and explicit (`slur_h`, `derog_dehum_h`, `derog_neg_attrib_h`, `profanity_h`) hate for a fine-tuned multilingual encoder (XLM-RoBERTa)?
- **SQ1.2:** How does this gap vary across the 11 languages in MHC [3]? Are certain languages disproportionately affected?

### RQ2: Can generating natural language explanations for hate labels improve classification accuracy, particularly for implicit hate?

- **SQ2.1:** Does a pipeline that jointly predicts a hate label and generates an explanation of the hateful intent (using a prompted LLM) outperform a classification-only baseline on implicit hate instances?
- **SQ2.2:** Does explanation generation provide greater gains for implicit hate (`derog_impl_h`) than for explicit categories, where lexical signals are already strong?
- **SQ2.3:** How does the quality of generated explanations (evaluated via human judgment or a reference-free metric) correlate with classification correctness?

### RQ3: Across multiple LLMs, does explanation-augmented prompting (vs. no explanation) improve robustness to the implicit–explicit distinction across languages?

- **SQ3.1:** How do multiple open-weight LLMs (e.g., Mistral-7B, Llama-3-8B, and the multilingual Aya-23 [9]) compare on implicit vs. explicit hate across MHC languages under a direct, no-explanation classification prompt?
- **SQ3.2:** For each LLM, does a structured explanation prompt (e.g., "Classify this text and explain why it is or is not hateful") outperform the no-explanation prompt, particularly on implicit hate (`derog_impl_h`)?
- **SQ3.3:** Is the benefit of explanation prompting consistent across models and languages, or is it model- and language-dependent? For which functionality–language combinations is the explanation effect largest or smallest?
- **SQ3.4:** Under few-shot prompting, does the *demonstration selection* strategy matter — e.g., does target-group / similarity-based selective retrieval (BM25 + adaptive threshold, as in ARIIHA [10]) improve implicit-hate detection over random shot selection, and does this interact with the explanation vs. no-explanation condition?

## Resources

- **Evaluation dataset:** Multilingual HateCheck (https://huggingface.co/datasets/mteb/multi-hatecheck) — 11 languages, diagnostic only.
- **Training data:** HASOC shared task datasets (overlapping MHC languages: English, German, Hindi) for the encoder baseline; **ToxiGen** [7] as an English implicit-hate reference corpus and as the template for synthetic generation.
- **Synthetic data generation:** **Aya-23 / Aya-101** (Cohere) [8, 9] and **Llama-3-8B** to generate ToxiGen-style implicit/non-hate examples in the MHC languages not covered by HASOC.
- **Encoder model:** XLM-RoBERTa (base or large).
- **LLMs (prompted, no fine-tuning):** Mistral-7B, Llama-3-8B, and the multilingual Aya-23 [9].

## Plan of Action

### Methods

**RQ1: Implicit vs. Explicit Performance Gap Across Languages.** XLM-RoBERTa (base or large) is fine-tuned on the available labelled corpus (HASOC for English/German/Hindi) for binary hate speech classification, then evaluated directly on MHC as a diagnostic benchmark. Per-functionality F1, precision, and recall are computed for the implicit category (`derog_impl_h`) and each explicit category across all 11 languages, producing a functionality × language results matrix from which the implicit–explicit gap and language-level disparities are quantified. Where the baseline reveals weak coverage (see the decision point in [implementation_plan.md](implementation_plan.md)), the training set is augmented with **synthetically generated** implicit examples in the missing MHC languages.

**RQ2: Explanation-Augmented Classification.** An open-weight LLM is prompted to jointly assign a hate label and generate a natural language rationale. Classification performance is compared against the XLM-RoBERTa baseline with separate breakdowns for implicit and explicit categories. Explanation quality is assessed via a reference-free metric (e.g., BERTScore or G-Eval), and its correlation with classification correctness examined.

**RQ3: Explanation vs. No-Explanation Across Multiple LLMs.** Multiple LLMs (Mistral-7B, Llama-3-8B, Aya-23) are prompted — without fine-tuning — across the full functionality × language matrix. For each model, two prompt conditions are contrasted (direct/no-explanation vs. structured explanation), and few-shot variants compare random vs. selective demonstration retrieval (BM25 + target-group, adaptive threshold [10]). Analysis focuses on whether the explanation effect generalises across models and on functionality–language combinations where models diverge most.

**Synthetic implicit-multilingual data generation.** Because HASOC covers only 3 MHC languages and carries no implicit labels, and ToxiGen is English-only, a massively multilingual LLM (**Aya-23/Aya-101** [8, 9], optionally Llama-3, or a closed frontier model such as Grok as a stronger but non-open alternative) generates ToxiGen-style implicit hate / matched non-hate pairs directly in the under-covered MHC languages, using demonstration-based prompting seeded from MHC functionality definitions. Generated data is quality-filtered and used to augment encoder fine-tuning (RQ1) and to populate the few-shot demonstration pool (RQ3).

### Workplan and task division

The execution plan (phases, parallel workstreams) and the parallelisable role division are maintained separately in [implementation_plan.md](implementation_plan.md). Roles there are left unassigned so members can select their workstream.

## References

[1] Mai ElSherief, Caleb Ziems, David Muchlinski, Vaishnavi Anber, Jordan Gregoire, Joleen Nham, and Diyi Yang. Latent hatred: A benchmark for understanding implicit hate speech. In *Proceedings of EMNLP 2021*, 2021.

[2] Faeze Ghorbanpour, Daryna Dementieva, and Alexander Fraser. Can prompting LLMs unlock hate speech detection across languages? A zero-shot and few-shot study. In *Proceedings of the 9th Workshop on Online Abuse and Harms (WOAH)*, Vienna, Austria, 2025. Association for Computational Linguistics.

[3] Paul Röttger, Haitham Seelawi, Debora Nozza, Zeerak Talat, and Bertie Vidgen. Multilingual HateCheck: Functional tests for multilingual hate speech detection models. In *Proceedings of the 6th Workshop on Online Abuse and Harms (WOAH)*, 2022.

[4] Paul Röttger, Bertie Vidgen, Dong Nguyen, Zeerak Waseem, Helen Margetts, and Janet Pierrehumbert. HateCheck: Functional tests for hate speech detection models. In *Proceedings of ACL-IJCNLP 2021*, pages 41–58, 2021.

[5] Manuel Tonneau, Diyi Liu, Niyati Malhotra, Scott A. Hale, Samuel Fraiberger, Victor Orozco-Olvera, and Paul Röttger. HateDay: Insights from a global hate speech dataset representative of a day on twitter. In *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 2297–2321, Vienna, Austria, 2025. Association for Computational Linguistics.

[6] Min Zhang, Jianfeng He, Taoran Ji, and Chang-Tien Lu. Don't go to extremes: Revealing the excessive sensitivity and calibration limitations of LLMs in implicit hate speech detection. In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 12073–12086, Bangkok, Thailand, 2024. Association for Computational Linguistics.

[7] Thomas Hartvigsen, Saadia Gabriel, Hamid Palangi, Maarten Sap, Dipankar Ray, and Ece Kamar. ToxiGen: A large-scale machine-generated dataset for adversarial and implicit hate speech detection. In *Proceedings of ACL 2022*, pages 3309–3326, 2022.

[8] Ahmet Üstün, Viraat Aryabumi, Zheng-Xin Yong, Wei-Yin Ko, Daniel D'souza, Gbemileke Onilude, et al. Aya Model: An instruction finetuned open-access multilingual language model. In *Proceedings of ACL 2024*, 2024. (arXiv:2402.07827)

[9] Viraat Aryabumi, John Dang, Dwarak Talupuru, Saurabh Dash, et al. Aya 23: Open weight releases to further multilingual progress. Cohere For AI, 2024. (arXiv:2405.15032)

[10] Selective demonstration retrieval for improved implicit hate speech detection (ARIIHA). 2025. (arXiv:2504.12082)
