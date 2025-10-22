# Paper 1 Pipeline Summary: FULL Mode

- **Run Seed**: 1234
- **Corpus Path**: `txt_corpus\W2V_Corpus.txt`
- **Corpus SHA-256**: `0aa89f31a3eb7fa49cdd875256e2ef18c70c57d99262263df661948bfd63212a`

## Model Hashes

- **Word2Vec CBOW SHA-256**: `a0fd14b9e94f1029acb346c7899ce146b66b5fd1a5fa935bfc65b33715170c31`
- **Word2Vec Skip-gram SHA-256**: `aeb291f0580801b0c3293c6d81deff81827cb77d9bff6977caf952913864222d`
- **Fine-Tuned BERT (dir hash)**: `88918a7a5280cadef5a4f785611a764050042cd3c579ef7283719e0cc99cb04e`

### ARI Scores: physical_vs_conceptual

| Model | ARI Score |
|-------|-----------|
| BERT Cased (Final) | 0.852 |
| BERT Cased (Final)KMeans+L2 | 0.852 |
| Word2Vec Skip-gramKMeans+L2 | 0.476 |
| BERT Cased (Final)[Agglo +L2] | 0.394 |
| Word2Vec CBOWKMeans+L2 | 0.376 |
| Word2Vec Skip-gram[Agglo +L2] | 0.376 |
| Word2Vec CBOW[Agglo +L2] | 0.208 |
| Word2Vec CBOW | 0.133 |
| Word2Vec Skip-gram | -0.109 |

### ARI Scores: tokonoma_as_conceptual

| Model | ARI Score |
|-------|-----------|
| BERT Cased (Final) | 0.719 |
| BERT Cased (Final)KMeans+L2 | 0.719 |
| Word2Vec Skip-gramKMeans+L2 | 0.373 |
| BERT Cased (Final)[Agglo +L2] | 0.304 |
| Word2Vec CBOWKMeans+L2 | 0.285 |
| Word2Vec Skip-gram[Agglo +L2] | 0.285 |
| Word2Vec CBOW[Agglo +L2] | 0.138 |
| Word2Vec CBOW | 0.100 |
| Word2Vec Skip-gram | -0.092 |

### WordSim Spearman Correlation

| Dataset | Model | Spearman ρ |
|---------|-------|------------|
| WordSim353 | Word2Vec CBOW | 0.167 |
| WordSim353_JA-EN | Word2Vec CBOW | 0.167 |
| WordSim353 | Word2Vec Skip-gram | 0.340 |
| WordSim353_JA-EN | Word2Vec Skip-gram | 0.340 |
| WordSim353 | BERT (Base) | 0.294 |
| WordSim353_JA-EN | BERT (Base) | 0.294 |
| WordSim353 | BERT (Fine-Tuned) | 0.317 |
| WordSim353_JA-EN | BERT (Fine-Tuned) | 0.317 |

## Figures

Figures for this run are in the `figures/` directory corresponding to seed 1234.
Note: Negative ARI values indicate agreement lower than chance (not an error).

