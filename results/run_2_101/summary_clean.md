# Paper 1 Pipeline Summary: CLEAN Mode

- **Run Seed**: 101
- **Corpus Path**: `txt_corpus\W2V_Corpus_CL.txt`
- **Corpus SHA-256**: `6edceac27f560cd4a869846f28a5371b40775beb84b6ca28e1307c82791f03fb`

## Model Hashes

- **Word2Vec CBOW SHA-256**: `a65ca587d6f7eebc241f3aa5897492be93252778e6cd557692973c229dd7a6b9`
- **Word2Vec Skip-gram SHA-256**: `61d2ddae79481f6110315d42e472fedae5eabd624bcb23aaeeb7bd005a355483`
- **Fine-Tuned BERT (dir hash)**: `4133af7e9478ed27734d1c5fb99ec99e865cef60f8b927450ecc0217f02f7b84`

### ARI Scores: physical_vs_conceptual

| Model | ARI Score |
|-------|-----------|
| BERT Cased (Final) | 0.718 |
| BERT Cased (Final)KMeans+L2 | 0.718 |
| BERT Cased (Final)[Agglo +L2] | 0.718 |
| Word2Vec Skip-gramKMeans+L2 | 0.476 |
| Word2Vec CBOW | 0.133 |
| Word2Vec CBOWKMeans+L2 | 0.077 |
| Word2Vec CBOW[Agglo +L2] | 0.025 |
| Word2Vec Skip-gram[Agglo +L2] | 0.025 |
| Word2Vec Skip-gram | -0.110 |

### ARI Scores: tokonoma_as_conceptual

| Model | ARI Score |
|-------|-----------|
| BERT Cased (Final) | 0.597 |
| BERT Cased (Final)KMeans+L2 | 0.597 |
| BERT Cased (Final)[Agglo +L2] | 0.597 |
| Word2Vec Skip-gramKMeans+L2 | 0.373 |
| Word2Vec CBOWKMeans+L2 | 0.134 |
| Word2Vec CBOW | 0.100 |
| Word2Vec CBOW[Agglo +L2] | 0.073 |
| Word2Vec Skip-gram[Agglo +L2] | -0.007 |
| Word2Vec Skip-gram | -0.095 |

### WordSim Spearman Correlation

| Dataset | Model | Spearman ρ |
|---------|-------|------------|
| WordSim353 | Word2Vec CBOW | 0.203 |
| WordSim353_JA-EN | Word2Vec CBOW | 0.203 |
| WordSim353 | Word2Vec Skip-gram | 0.319 |
| WordSim353_JA-EN | Word2Vec Skip-gram | 0.319 |
| WordSim353 | BERT (Base) | 0.294 |
| WordSim353_JA-EN | BERT (Base) | 0.294 |
| WordSim353 | BERT (Fine-Tuned) | 0.378 |
| WordSim353_JA-EN | BERT (Fine-Tuned) | 0.378 |

## Figures

Figures for this run are in the `figures/` directory corresponding to seed 101.
Note: Negative ARI values indicate agreement lower than chance (not an error).

