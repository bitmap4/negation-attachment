# Results, 16 November 2026

Model: Qwen3-4B, no examples in the prompt, float16, one RTX 2080 Ti. All numbers are from the files in this folder.
Splits: the sites and subspaces come from `disc`; DAS and E1 numbers are on `test` (other names and objects).

## Decisions that differ from the slides

| Slide | Now | Reason |
| --- | --- | --- |
| Qwen3-1.7B | Qwen3-4B | 1.7B gets 62% in Hindi with no examples (gate: 80%). 4B gets 100%. |
| 448 heads | 1152 heads (36 layers x 32) | Model change. |
| Oracle for 1.7B | `adamkarvonen/checkpoints_latentqa_cls_past_lens_Qwen3-4B` | Model change. It reads layers 9, 18, 27. |

Qwen3-1.7B with fixed examples also passes in Hindi (2 examples: 99%), and its French route is the same (see E2).

## E1, behaviour

| Set | Hindi | French |
| --- | --- | --- |
| Base task, dev and test, 19,720 items for each language | 100.0% | 100.0% |
| Affirmative controls (1,000) | 100% | 100% |
| hab, never, no longer, nothing, lexical (all conditions) | 100% | 99.9-100% |
| Focus on the object / on the person | 97.9% / 88.4% | 99.8% / 100% |
| Worlds: w1, w2 | 100%, 99.5-99.8% | 100%, 95.3-97.0% |
| Nested negation | 99.2-100% | 92.5-97.1% |
| Quantifiers, all conditions but one | 92-100% | 97-100% |

## E2, where is the attachment

Residual stream patched from the minimal partner, positions aligned by role. Recovery of the logit difference, 96 pairs.

| Stage | Layers | Evidence (Hindi / French) |
| --- | --- | --- |
| Tokens of both clauses | 7-15 | Both clauses: 0.97 / 0.75 at layer 13. One clause alone: at most 0.25 / 0.19. |
| Option names learn which mention they name | 12-14 | Names exchanged in the context: option tokens recover 0.80 / 0.98 at layer 14. |
| Question polarity on the question tokens | 15-18 | Other question polarity: question tokens 0.54 / 0.71 at layer 15; question and options together 0.94 / 0.94 at layer 17. |
| Option tokens (lookup at the options, H1) | 18-20 | Options: 0.91 / 0.91 at layer 19. Option A alone: 0.36 / 0.39. |
| "B:" label and template tokens | 20-24 | "B:" 0.14-0.34. |
| Last position | 25+ | 0.91 / 0.90 at layer 25. |

Controls:
- Donor with other names and another object: the context state still transfers (0.95 / 0.86 at layer 15). So the context code is abstract.
- Donor with the option order changed: context patch still recovers 1.00 / 0.95 (layer 15); option patch recovers 0.07 / 0.11 at layer 19. So from layer 19 the option tokens hold the answer letter.
- One attention layer or one MLP alone recovers little, except attention 18 (0.68, Hindi) and attention 24 at the last position (0.66).

Heads (ranked on 32 pairs):

| Step | Layers | Top 8 heads together | All other heads of these layers | Best single head |
| --- | --- | --- | --- | --- |
| To the option tokens | 16-19 | 0.90 / 0.88 | 0.03 / 0.05 | 0.02 / 0.04 |
| To "B:" and template | 19-24 | 0.53 / 0.57 (top 64: 0.74 / 0.95) | 0.02 / 0.03 | 0.03 / 0.04 |
| To the last position | 23-26 | 0.83 / 0.82 | -0.07 / -0.05 | 0.16 / 0.10 |

The top 8 sets are nearly the same in both languages: 7 shared heads for the options (L16H17, L17H4, L18H8, L18H11, L18H15, L19H10, L19H31), 7 for "B:", 6 for the last position (among them L24H28-31).
Attention from an option name goes to the clause of the same person: L13H13 0.98 own clause against 0.01 (Hindi), 0.89 against 0.04 (French). L11H10 and L19H24 look at the name; L13H13, L13H14, L15H13, L18H15 look at the predicate.

Negation forms: the late sites are the same for all forms. Options at layer 20: 0.79-0.91. Last position at layer 26: 0.87-0.93. The context stage depends on the form (both clauses at layer 14: 0.04-0.98).

Qwen3-1.7B, French, no examples: the same order of stages: context to layer 11 (0.74), options at 14-15 (0.97), "B:" at 16-21, last position from 22.

## E3, causal variables (DAS)

Interchange accuracy on pairs where the variable differs, 8 dimensions, train and test in the same language (Hindi / French).
W: the negated mention. Z: option A names the negated person. ANS: the answer letter.

| Site | W | Z | ANS |
| --- | --- | --- | --- |
| Both clauses, layer 14 | **0.92 / 0.81** | 0.17 / 0.03 | 0.33 / 0.09 |
| Two option names, layer 18 | 0.07 / 0.25 | **0.75 / 0.90** | 0.23 / 0.66 |
| Two option names, layer 19 | 0.12 / 0.34 | 0.60 / 0.56 | **0.93 / 0.97** |
| Last position, layer 27 | 0.01 / 0.10 | 0.38 / 0.45 | **1.00 / 1.00** |

- Random subspaces: 0.00 everywhere.
- Affirmative controls keep their answer under the W and Z subspaces at these sites (0.99-1.00). They change under the ANS subspace (0.5), as the model predicts.
- Dimensions: ANS needs 1 (last position 1.00; options 0.81 / 0.96). W in the context: 0.59 / 0.23 with 1, 0.72 / 0.74 with 4.

Transfer between languages (subspace trained on the first language):

| Subspace | Hindi to French | French to Hindi | Base in the training language, source in the other |
| --- | --- | --- | --- |
| W, both clauses, layer 14 | 0.10 | 0.49 | 0.37 / 0.91 |
| Z, options, layer 18 | 0.61 | 0.40 | 0.72 / 0.83 |
| ANS, options, layer 19 | 0.98 | 0.45 | 0.97 / 0.93 |
| ANS, last position, layer 27 | 1.00 | 0.98 | 1.00 / 0.96 |

Frozen subspaces on the negation forms: see `fig/e3_forms.png`. The answer subspaces transfer to all forms except focus (options: 0.80-0.99; last position: 0.99-1.00). The W subspace transfers to habitual negation and "never" (0.63-0.95), and to "no longer" in Hindi only (0.90 against 0.03 in French).

## E3 on the explicit worlds (slide 9)

Patching, two persons and two predicates (64 items): a change of the queried fact changes the answer; a change of another fact (same person, or other person) has an effect below 0.04 at every layer and position, in both languages.

DAS, one subspace for all facts, site = all tokens of the fact after the name:

| Data | Target, same slot | Target, other slot | Other facts stay correct |
| --- | --- | --- | --- |
| Hindi, two predicates, layer 13 | 0.99 | 0.97 | 0.99 |
| Hindi, one predicate, layer 15 | 0.91 | 0.87 | 1.00 |
| French, one predicate, layer 15 | 0.74 | 0.85 | 0.99-1.00 |
| French, two predicates, layer 15 | 0.62 | 0.54 | 0.97-0.98 |

Nested negation ("it is not true that ..."): at layer 11 the fact tokens hold the polarity of the inner clause (Hindi: 0.99 for the inner-clause model, 0.45 for the composed model). In French at layer 13 the composed polarity fits better (0.91 against 0.64).

## E4, activation oracle

Readout, 300 test items for each language (Hindi / French):

| Question, states given to the oracle | Oracle | Adapter off | Linear probe |
| --- | --- | --- | --- |
| "Does this person like it?", clause 1 tokens, layer 18 | 0.99 / 1.00 | 0.57 / 0.64 | 1.00 |
| "Does <second name> like it?", all context tokens, layer 18 | 1.00 / 1.00 | 0.50 | 1.00 |
| "Does this person like it?", option A name, layer 18 | 0.66 / 0.57 (AUC 0.75 / 0.68) | 0.42 / 0.43 | 1.00 |
| "Is the negation about the first person?", context, layer 18 | 0.50 / 0.50 | 0.50 | 1.00 |
| "Is the correct answer option A?", last position, layer 27 | 0.95 / 0.96 | 0.55 / 0.54 | 1.00 |

After an intervention (96 pairs): the oracle question about the answer, on the last position at layer 27.

| Patch | Model answer changed | Oracle agrees with the new answer |
| --- | --- | --- |
| None | 0% | 0.94 / 0.95 |
| Both clauses, layer 13 | 100% / 73% | 0.94 / 0.93 |
| Clause 1 only, layer 13 | 1% / 4% | 0.92 / 0.97 |
| Clause 2 only, layer 13 | 10% / 7% | 0.91 / 0.95 |
| Option tokens, layer 19 | 98% / 95% | 0.76 / 0.92 |

## Limits

- One model for the full analysis. One seed for each DAS run.
- Recovery is not additive: one head, one layer or one clause alone does little.
- The head ranking used 32 pairs.
- The worlds DAS has one seed.
- The new Hindi and French templates are not yet checked by a fluent reader (`data/templates.md`).

## Figures (`fig/`)

`e2_heat.png`, `e2_flow.png`, `e2_heads.png`, `e2_worlds.png`, `e3_das.png`, `e3_transfer.png`, `e3_forms.png`, `e4_oracle.png`.
## To do

- Sets of heads on the test split, random head sets, resample ablation
- The test of H3: mean ablation of the late heads, for affirmative and negative questions
- DAS seeds for the main sites and for the worlds
- Patching between two quantifier sentences
- The word order of the Hindi exception phrase
- One minimal pair with the main patches, for the presentation
