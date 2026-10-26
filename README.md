# Negation attachment in Hindi and French

Course project for CL3.412 Mechanistic Interpretability (IIIT Hyderabad), by Ishaan Romil and Abhyudit Singh.

Each item has two persons and one negation, for example "Aman does not like curd, but Arjun likes curd. Who likes curd?".
A minimal partner has the same words with the negation in the other clause. We study how the model attaches the negation
to the correct person in Hindi and French, and we start with the behaviour: does it answer both members of a pair correctly?

## Requirements

Python 3.12, `torch`, `transformers` (4.53 or later), `peft` and `numpy`.
`jobs/setup_env.sh` builds the environment that we used.

```bash
python data/gen.py data             # write the base data sets again (they are in data/)
python -m neg.e1_behaviour results/e1.json hi_dev fr_dev
```

## Settings (environment variables)

| Variable | Default | Use |
| --- | --- | --- |
| `NEG_MODEL` | `Qwen/Qwen3-1.7B` | Model name. |
| `NEG_SHOTS` | `0` | Number of fixed examples before the item (0 to 4). |

## Layout

| Path | Content |
| --- | --- |
| `data/gen.py` | Base task: two persons, one negation that moves, A/B answer. |
| `data/shots.py` | Four fixed examples for each language (used only with `NEG_SHOTS`). |
| `neg/common.py` | Prompt format, role label for each token, model, hooks. |
| `neg/e1_behaviour.py` | E1: accuracy and logit difference for each condition. |

## Data sets

`data/<lang>_<split>.jsonl.gz` is the base task (splits `disc`, `dev`, `test`).
Each item has `roles`: the role of each character span. `common.token_roles` gives one role to each token.
