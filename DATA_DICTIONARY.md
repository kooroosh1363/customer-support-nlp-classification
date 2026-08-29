# Data Dictionary

| Field | Meaning | Used by model? |
| --- | --- | --- |
| `text` | Customer utterance / support request text | Yes |
| `category` | One of 77 Banking77 intent labels | Target only |

The project does not add customer identity, account numbers, or synthetic metadata. Model input is the text itself. Labels are operational intent categories such as card, cash-withdrawal, transfer, payment, cash-deposit, and account-related intents defined by the benchmark.

Exact-text overlap between train/validation and train/test is audited in the pipeline because duplicate utterances across splits can inflate NLP evaluation.
