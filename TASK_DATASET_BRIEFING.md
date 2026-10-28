# Task and situation dataset briefing

The task study has passed feature creation and semantic consolidation and now has substantial independent applicability data. We can inspect it while human-item inference continues.

| Stage | Current task checkpoint |
|---|---|
| Generation | 131 of 135 generation jobs accepted across the 15 items, three prompt families and three repetitions. Four missing samples remain explicit. |
| Candidate inventory | 875 accepted task phrases map to 592 canonical feature IDs in the shared 5,825-feature bank. |
| Applicability | 87,375 item–feature cells: 69,341 resolved (79.4%), comprising 11,453 yes and 57,888 no; 18,034 unresolved. |
| Composition | Component/direct discovery comparisons, sampling allocations, applicability gains and provisional interaction candidates are computed. Missing samples and unresolved cells limit final claims. |
| Interpretation | Labels remain automated and unreviewed. Semantic theme clusters and a feature ontology have not yet been assigned. |

Generate the task-only figures offline:

```bash
MPLCONFIGDIR=artifacts/feature_space_full/matplotlib_cache .venv/bin/python -m feature_space_pilot.task_focus.run
```

`artifacts/feature_space_task_focus/latest.json` points to an immutable export with PNG/SVG figures, every underlying task applicability cell and original task feature provenance.

Start with `figure_01_task_coverage`, then `figure_02_task_feature_map`. The map places actual propositions against both task hierarchies and the independent situation axis. Its 30 display examples do not cap the analysis vocabulary: all 87,375 cells remain in `task_applicability.csv`. `task_discovered_features.csv` lists all 592 task-discovered canonical features.

`figure_03_coapplicability_patterns` groups identical binary profiles. Of the full vocabulary, 677 features are resolved across all 15 items: 254 have at least one positive value and 423 are zero everywhere. The other 5,148 features have unresolved cells. Consequently, this is a restricted complete-case preview. Identical profiles do not establish equivalent meanings or validate the underlying labels. Every group member is exported.

The export also contains 86 parent-positive/situated-child-negative task-origin cells in `inheritance_review_flags.csv`. These are screening flags, not 86 confirmed errors: situations can legitimately change some properties. One review example is “lines 1 and 3 must rhyme,” marked 1 for `poem_T` and 0 for `poem_S_deadline`, although both retain the ABAB CDCD requirement. Inspect unchanged explicit requirements before interpreting such differences as situation effects. No label has been corrected automatically.

A useful next layer is a multi-label feature taxonomy: goals/outputs, operations/information, format/quality constraints, time/resources, evaluation/social context, and relations/trade-offs. Theme and abstraction should be separate metadata. The existing item hierarchy is domain → task type → instance; situations are an independent axis. Feature granularity could range from output constraints to rhyme constraints to the concrete ABAB CDCD requirement. Multiple tags may apply while each original binary column remains available.

Distinguish semantic grouping of proposition text from grouping by applicability pattern. Jaccard distance with average linkage can support descriptive clustering of binary profiles after missingness is handled transparently. Text embeddings and provisional model labels can propose semantic families, with representative features audited. Neither method automatically establishes equivalence or a parent–child ontology. Similarity matrices and multidimensional scaling can summarize item relationships once adequate judgments are available. Set-intersection plots can compare component and direct discoveries, preserving sampling gaps and uncertainty.

Research precedents:

- [Harpaintner, Trumpp & Kiefer (2018)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6156367/) coded properties into categories and clustered concepts by category profiles, using dendrograms and cluster-profile plots. Their human categories should not be transferred to these tasks without a declared coding scheme.
- [Kriegeskorte, Mur & Bandettini (2008)](https://www.frontiersin.org/journals/systems-neuroscience/articles/10.3389/neuro.06.004.2008/full) describe representational dissimilarity matrices and multidimensional scaling for comparing representations.
- [Lex et al. (2014), UpSet](https://vdl.sci.utah.edu/publications/2014_infovis_upset/) use matrix-based set-intersection displays, useful for comparing task, situation and combined discovery sets.

My priority recommendation is to inspect actual task feature maps, audit uncertain or contradictory labels, and tag the 592 task-discovered features by theme and scope. Composition plots and task geometry can then support more interpretable conclusions. The briefing and export make no API calls and leave active inference and original experiments intact.
