# Paper 1 Pipeline Summary: FULL Mode

- **Run Seed**: 527
- **Corpus Path**: `txt_corpus\W2V_Corpus.txt`
- **Corpus SHA-256**: `0aa89f31a3eb7fa49cdd875256e2ef18c70c57d99262263df661948bfd63212a`

## Model Hashes

- **Word2Vec CBOW SHA-256**: `08f193b1e44fde7f3f8ed45fb91edfe734692c3fe6d9a63c730040700ed30676`
- **Word2Vec Skip-gram SHA-256**: `0d045f639d5f3645a5b86e1b7d64a35220b86e1b7e25adcc8d8b9f51af8f7bd5`
- **Fine-Tuned BERT (dir hash)**: `46b41feec909eab5d5338214bc782094d0b6178887d36610bd82c38f4a58b7a2`

### ARI Scores: physical_vs_conceptual

| Model | ARI Score |
|-------|-----------|
| BERT Cased (Final) | 0.852 |
| BERT Cased (Final)KMeans+L2 | 0.852 |
| Word2Vec Skip-gram[Agglo +L2] | 0.476 |
| BERT Cased (Final)[Agglo +L2] | 0.394 |
| Word2Vec Skip-gramKMeans+L2 | 0.376 |
| Word2Vec CBOWKMeans+L2 | 0.208 |
| Word2Vec CBOW | 0.133 |
| Word2Vec CBOW[Agglo +L2] | -0.100 |
| Word2Vec Skip-gram | -0.109 |

### ARI Scores: tokonoma_as_conceptual

| Model | ARI Score |
|-------|-----------|
| BERT Cased (Final) | 0.719 |
| BERT Cased (Final)KMeans+L2 | 0.719 |
| Word2Vec Skip-gram[Agglo +L2] | 0.373 |
| BERT Cased (Final)[Agglo +L2] | 0.304 |
| Word2Vec Skip-gramKMeans+L2 | 0.285 |
| Word2Vec CBOWKMeans+L2 | 0.138 |
| Word2Vec CBOW | 0.100 |
| Word2Vec CBOW[Agglo +L2] | -0.066 |
| Word2Vec Skip-gram | -0.092 |

### WordSim Spearman Correlation

| Dataset | Model | Spearman ρ |
|---------|-------|------------|
| WordSim353 | Word2Vec CBOW | 0.199 |
| WordSim353_JA-EN | Word2Vec CBOW | 0.199 |
| WordSim353 | Word2Vec Skip-gram | 0.337 |
| WordSim353_JA-EN | Word2Vec Skip-gram | 0.337 |
| WordSim353 | BERT (Base) | 0.294 |
| WordSim353_JA-EN | BERT (Base) | 0.294 |
| WordSim353 | BERT (Fine-Tuned) | 0.344 |
| WordSim353_JA-EN | BERT (Fine-Tuned) | 0.344 |

## Figures

Figures for this run are in the `figures/` directory corresponding to seed 527.
Note: Negative ARI values indicate agreement lower than chance (not an error).

