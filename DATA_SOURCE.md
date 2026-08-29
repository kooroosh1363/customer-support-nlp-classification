# Data Source

Dataset: **Banking77**, a public intent-classification benchmark released by PolyAI.

Repository source used by this project:
`PolyAI-LDN/task-specific-datasets/banking_data`

Files consumed directly by CI:
- `train.csv`
- `test.csv`
- `categories.json` for source reference

The upstream repository exposes 77 customer-service banking intents and separate official train/test CSV files. This project keeps the upstream test file locked as the final test set and creates a stratified validation split only from the official training file.

The pipeline caches source CSVs under `data/raw/` after download. Raw data and generated artifacts are not committed.

Important claim boundary: Banking77 is a benchmark for banking customer-service intent classification. It is used here to demonstrate a support-routing NLP system; it is not evidence of performance on every company's support taxonomy or live ticket distribution.
