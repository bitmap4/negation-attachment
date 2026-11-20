# Negation attachment in Hindi and French

Course project for CL3.412 Mechanistic Interpretability (IIIT Hyderabad), by Ishaan Romil and Abhyudit Singh.

Each item has two persons and one negation, for example "Aman does not like curd, but Arjun likes curd. Who likes curd?".
A minimal partner has the same words with the negation in the other clause. We study how Qwen3-4B attaches the negation
to the correct person in Hindi and French: behaviour (E1), activation patching (E2), distributed alignment search (E3)
and an activation oracle (E4).

The report is `report/main.pdf`. The numbers are in `results/RESULTS.md` and the figures in `results/fig/`.

## Requirements

Python 3.12, `torch`, `transformers` (4.53 or later), `peft`, `numpy`, and `matplotlib` for the figures.
`jobs/setup_env.sh` builds the environment that we used. Qwen3-4B in 16-bit floats needs a GPU with 11 GB.

```bash
python data/gen.py data && python data/gen2.py      # write the data sets again (they are in data/)
NEG_MODEL=Qwen/Qwen3-4B NEG_DTYPE=half python -m neg.e1_behaviour results/e1.json hi_test fr_test
NEG_MODEL=Qwen/Qwen3-4B NEG_DTYPE=half python -m neg.e2_patch results/resid_hi.npz --lang hi --mode neg --kind resid
python -m neg.figs                                  # figures from the files in results/
```

`jobs/cmds/*.txt` are the command lists of all runs.

## Layout

| Path | Content |
| --- | --- |
| `data/gen.py` | Base task: two persons, one negation that moves, A/B answer. |
| `data/gen2.py` | Negation forms, scope sets and explicit worlds. Writes `data/templates.md`. |
| `data/shots.py` | Four fixed examples for each language (used only with `NEG_SHOTS`). |
| `neg/common.py` | Prompt format, role label for each token, model, hooks. |
| `neg/e1_behaviour.py` | E1: accuracy and logit difference for each condition. |
| `neg/e2_patch.py` | E2: activation patching (residual, attention, MLP, heads) with role alignment. |
| `neg/e3_das.py` | E3: DAS for the negated mention, with transfer between languages. |
| `neg/e4_oracle.py` | E4: activation oracle readout, with two baselines. |
| `neg/pairs.py` | Recipient, partner and donor items for patching. |
| `neg/e2_heads.py` | E2: head ranking and sets of heads. |
| `neg/e2_attn.py` | E2: attention patterns from the option names and the last position. |
| `neg/e2w_patch.py` | E2 on the explicit worlds (target fact and other facts). |
| `neg/e3w_das.py` | E3 on the explicit worlds: polarity for each fact, target and collateral scores. |
| `neg/e2_ablate.py` | E2: head sets on the test split, random heads, resample ablation, and the test of H3 (shortcut heads). |
| `neg/e2q_patch.py` | E2 on the quantifier set: patching between two quantifier sentences. |
| `neg/e1_except.py` | E1: the Hindi "nobody except X" items with other wordings of the exception. |
| `neg/figs.py` | All figures (`results/fig/`). |
| `report/` | The report (`main.pdf`). Build: `latexmk -lualatex main.tex`. |
| `results/RESULTS.md` | Summary of the results with the numbers. |
| `jobs/` | Slurm scripts for ada. `jobs/cmds/*.txt` are the command lists that were run. |
| `results/` | Outputs copied from ada. |

## Settings (environment variables)

| Variable | Default | Use |
| --- | --- | --- |
| `NEG_MODEL` | `Qwen/Qwen3-1.7B` | Model name. The main results use `Qwen/Qwen3-4B` with `NEG_DTYPE=half`. |
| `NEG_DTYPE` | `float32` | `half` is necessary for Qwen3-4B on an 11 GB GPU. |
| `NEG_SHOTS` | `0` | Number of fixed examples before the item (0 to 4). |

## Run on ada

```bash
rsync -a --exclude results --exclude logs ./ ada:cursor/negation/
# on ada, in ~/cursor/negation
sbatch --gres=gpu:2 --cpus-per-task=8 --mem=48G -J name jobs/multi.sh jobs/cmds/<file>.txt
```

`jobs/multi.sh` builds the environment on the node (`/scratch/$USER/venvs/negation`), refuses GPUs
older than the RTX 2080 Ti, and divides the command lines between the GPUs of the job.
Only partition `u22` is open to our account, so the jobs use `--constraint=2080ti`.

## Data sets

`data/<lang>_<split>.jsonl.gz` is the base task (splits `disc`, `dev`, `test`).
`data/<lang>_<set>_<split>.jsonl.gz` are the new sets:

| Set | Content | Answer options |
| --- | --- | --- |
| `hab` | Sentential negation with habitual verbs | two names |
| `never` | ne ... jamais / कभी ... नहीं | two names |
| `nolonger` | ne ... plus / अब ... नहीं, with a time context | two names |
| `nothing` | ne ... rien / कुछ नहीं | two names |
| `lex` | absent against not present, in the context and in the question | two names |
| `focus` | The negation targets one of two objects, or one of two persons | objects or names |
| `quant` | everyone / nobody, with an exception; "not everyone" with facts | yes / no |
| `w1` | Two persons, one predicate, four polarity states | yes / no |
| `nest` | `w1` with "it is not true that" around the queried fact | yes / no |
| `w2` | Two persons, two predicates, sixteen polarity states | yes / no |

Each item has `roles`: the role of each character span. `common.token_roles` gives one role to each token.
