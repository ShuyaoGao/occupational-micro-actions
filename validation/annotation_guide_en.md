# Annotation Guide (English; translated from the Chinese original given to annotators, which is retained by the authors and available to editors on request)

> Purpose: independent quality verification for a study on "AI and occupational substitution."
> **Please work independently — do not discuss with the other annotators and do not consult any AI
> for answers**; we want exactly your unaided judgement. Each of the three annotators fills in
> the file bearing their own number.

You will receive two tables (CSV, openable in Excel): **A: decomposition quality**, **B: action
types**. About 1.5-2 hours.

## Task A: decomposition quality (sampleA_..., ~79 items)
Each row is an occupational work activity (DWA_title) and the sequence of "micro-actions" it was
decomposed into (pipe-separated). Rate the decomposition on three 1-5 scales:
- **Q1 completeness**: does the sequence cover the activity's main phases? 5 = no obvious
  omissions; 1 = key steps missing.
- **Q2 faithfulness (no hallucination)**: does every step truly belong to this activity?
  5 = all relevant, nothing fabricated; 1 = clearly extraneous steps present.
- **Q3 atomic granularity**: is the step grain reasonable (neither one-step-equals-the-whole
  activity nor absurdly fragmented)? 5 = appropriate; 1 = far too coarse or too fine.
If unsure, go with your intuition; one sentence in `notes` is welcome. You are NOT asked to
re-decompose anything - only to judge the given decomposition.

## Task B: action-type blind labelling (sampleB_..., 150 items)
Each row is one micro-action description. Assign it to exactly one of four classes in `label`:
- **L = Linguistic**: reading/writing, calculation, processing text or symbolic information.
- **MP = Multimodal Perception**: visual inspection, monitoring instruments, perceiving scenes/sounds.
- **E = Embodied**: physical manipulation, using tools, bodily movement.
- **HB = Human-Bound**: interpersonal trust, emotional support, negotiation, accountable judgement.
Choose the SINGLE dominant requirement of the step.

---
