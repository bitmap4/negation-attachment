# Negation attachment in Hindi and French

Course project for CL3.412 Mechanistic Interpretability (IIIT Hyderabad), by Ishaan Romil and Abhyudit Singh.

Each item has two persons and one negation, for example "Aman does not like curd, but Arjun likes curd. Who likes curd?".
A minimal partner has the same words with the negation in the other clause. We study how the model attaches the negation
to the correct person in Hindi and French, and we start with the behaviour: does it answer both members of a pair correctly?

## Requirements

Python 3.12, `torch`, `transformers` (4.53 or later), `peft` and `numpy`.

```bash
python data/gen.py data             # write the base data sets again (they are in data/)
```

## Layout

| Path | Content |
| --- | --- |
| `data/gen.py` | Base task: two persons, one negation that moves, A/B answer. |
| `data/shots.py` | Four fixed examples for each language (used only with `NEG_SHOTS`). |

## Data sets

`data/<lang>_<split>.jsonl.gz` is the base task (splits `disc`, `dev`, `test`).
Each item has `roles`: the role of each character span.
