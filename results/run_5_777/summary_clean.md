# Paper 1 Pipeline Summary: CLEAN Mode

- **Run Seed**: 777
- **Corpus Path**: `txt_corpus\W2V_Corpus_CL.txt`
- **Corpus SHA-256**: `6edceac27f560cd4a869846f28a5371b40775beb84b6ca28e1307c82791f03fb`

## Model Hashes

- **Word2Vec CBOW SHA-256**: `26c9e4f62d7eca68c2e067e4f75fad97a9710427acfa40999040e061836e683e`
- **Word2Vec Skip-gram SHA-256**: `935e9b410b5c90b1b5632c8457704949ec4837602cada4b4c435fbc786682892`
- **Fine-Tuned BERT (dir hash)**: `de3cc5a4b0a353dd8af2a4883430434f4a872210410c483e24adc467433b3322`

### ARI Scores: physical_vs_conceptual

| Model | ARI Score |
|-------|-----------|
| BERT Cased (Final) | 0.718 |
| BERT Cased (Final)KMeans+L2 | 0.718 |
| BERT Cased (Final)[Agglo +L2] | 0.718 |
| Word2Vec Skip-gramKMeans+L2 | 0.476 |
| Word2Vec CBOW | 0.133 |
| Word2Vec Skip-gram[Agglo +L2] | 0.025 |
| Word2Vec CBOWKMeans+L2 | -0.036 |
| Word2Vec CBOW[Agglo +L2] | -0.081 |
| Word2Vec Skip-gram | -0.109 |

### ARI Scores: tokonoma_as_conceptual

| Model | ARI Score |
|-------|-----------|
| BERT Cased (Final) | 0.597 |
| BERT Cased (Final)KMeans+L2 | 0.597 |
| BERT Cased (Final)[Agglo +L2] | 0.597 |
| Word2Vec Skip-gramKMeans+L2 | 0.373 |
| Word2Vec CBOW | 0.100 |
| Word2Vec Skip-gram[Agglo +L2] | -0.007 |
| Word2Vec CBOWKMeans+L2 | -0.037 |
| Word2Vec CBOW[Agglo +L2] | -0.066 |
| Word2Vec Skip-gram | -0.092 |

### WordSim Spearman Correlation

| Dataset | Model | Spearman ρ |
|---------|-------|------------|
| WordSim353 | Word2Vec CBOW | 0.170 |
| WordSim353_JA-EN | Word2Vec CBOW | 0.170 |
| WordSim353 | Word2Vec Skip-gram | 0.336 |
| WordSim353_JA-EN | Word2Vec Skip-gram | 0.336 |
| WordSim353 | BERT (Base) | 0.294 |
| WordSim353_JA-EN | BERT (Base) | 0.294 |
| WordSim353 | BERT (Fine-Tuned) | 0.326 |
| WordSim353_JA-EN | BERT (Fine-Tuned) | 0.326 |

## Figures

Figures for this run are in the `figures/` directory corresponding to seed 777.
Note: Negative ARI values indicate agreement lower than chance (not an error).

