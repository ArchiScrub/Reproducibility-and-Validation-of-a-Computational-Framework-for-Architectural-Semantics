# Paper 1 Pipeline Summary: CLEAN Mode

- **Run Seed**: 42
- **Corpus Path**: `txt_corpus\W2V_Corpus_CL.txt`
- **Corpus SHA-256**: `6edceac27f560cd4a869846f28a5371b40775beb84b6ca28e1307c82791f03fb`

## Model Hashes

- **Word2Vec CBOW SHA-256**: `845869822712e8f0951841f75c63d4874d6b7d2280f83ab93da4769ebde36bb6`
- **Word2Vec Skip-gram SHA-256**: `3203c736c8f58d59a1c3d1d95df3493c3729b38e351958202c8c3e92f584de57`
- **Fine-Tuned BERT (dir hash)**: `5b0aa1ce5c2e27f9516e23766e1c13ef0536affb78a4e822eb02a8a0cfa1e119`

### ARI Scores: physical_vs_conceptual

| Model | ARI Score |
|-------|-----------|
| BERT Cased (Final) | 0.718 |
| BERT Cased (Final)KMeans+L2 | 0.718 |
| BERT Cased (Final)[Agglo +L2] | 0.718 |
| Word2Vec Skip-gramKMeans+L2 | 0.476 |
| Word2Vec Skip-gram[Agglo +L2] | 0.476 |
| Word2Vec CBOW | 0.133 |
| Word2Vec CBOW[Agglo +L2] | -0.019 |
| Word2Vec CBOWKMeans+L2 | -0.081 |
| Word2Vec Skip-gram | -0.109 |

### ARI Scores: tokonoma_as_conceptual

| Model | ARI Score |
|-------|-----------|
| BERT Cased (Final) | 0.597 |
| BERT Cased (Final)KMeans+L2 | 0.597 |
| BERT Cased (Final)[Agglo +L2] | 0.597 |
| Word2Vec Skip-gramKMeans+L2 | 0.373 |
| Word2Vec Skip-gram[Agglo +L2] | 0.373 |
| Word2Vec CBOW | 0.100 |
| Word2Vec CBOW[Agglo +L2] | 0.021 |
| Word2Vec CBOWKMeans+L2 | -0.054 |
| Word2Vec Skip-gram | -0.092 |

### WordSim Spearman Correlation

| Dataset | Model | Spearman ρ |
|---------|-------|------------|
| WordSim353 | Word2Vec CBOW | 0.181 |
| WordSim353_JA-EN | Word2Vec CBOW | 0.181 |
| WordSim353 | Word2Vec Skip-gram | 0.328 |
| WordSim353_JA-EN | Word2Vec Skip-gram | 0.328 |
| WordSim353 | BERT (Base) | 0.294 |
| WordSim353_JA-EN | BERT (Base) | 0.294 |
| WordSim353 | BERT (Fine-Tuned) | 0.341 |
| WordSim353_JA-EN | BERT (Fine-Tuned) | 0.341 |

## Figures

Figures for this run are in the `figures/` directory corresponding to seed 42.
Note: Negative ARI values indicate agreement lower than chance (not an error).

