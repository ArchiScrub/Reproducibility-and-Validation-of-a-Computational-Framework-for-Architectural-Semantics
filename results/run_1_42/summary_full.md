# Paper 1 Pipeline Summary: FULL Mode

- **Run Seed**: 42
- **Corpus Path**: `txt_corpus\W2V_Corpus.txt`
- **Corpus SHA-256**: `0aa89f31a3eb7fa49cdd875256e2ef18c70c57d99262263df661948bfd63212a`

## Model Hashes

- **Word2Vec CBOW SHA-256**: `6cbcaba90a7aa5dba52b370475e22df6e0d5fbc7bc8341beea1d846496cce697`
- **Word2Vec Skip-gram SHA-256**: `14588d5c8e349eecce3c786cf2daf03de11460302d95b4aa64fdbd715aac3e78`
- **Fine-Tuned BERT (dir hash)**: `1a7537968edaf5f61c3af34911e92e7f5662ba956d0b1fabf2884bdca8cdf827`

### ARI Scores: physical_vs_conceptual

| Model | ARI Score |
|-------|-----------|
| BERT Cased (Final) | 0.852 |
| BERT Cased (Final)KMeans+L2 | 0.852 |
| BERT Cased (Final)[Agglo +L2] | 0.394 |
| Word2Vec CBOW | 0.133 |
| Word2Vec CBOWKMeans+L2 | -0.032 |
| Word2Vec Skip-gramKMeans+L2 | -0.081 |
| Word2Vec Skip-gram[Agglo +L2] | -0.081 |
| Word2Vec CBOW[Agglo +L2] | -0.100 |
| Word2Vec Skip-gram | -0.109 |

### ARI Scores: tokonoma_as_conceptual

| Model | ARI Score |
|-------|-----------|
| BERT Cased (Final) | 0.719 |
| BERT Cased (Final)KMeans+L2 | 0.719 |
| BERT Cased (Final)[Agglo +L2] | 0.304 |
| Word2Vec CBOW | 0.100 |
| Word2Vec CBOWKMeans+L2 | -0.041 |
| Word2Vec CBOW[Agglo +L2] | -0.066 |
| Word2Vec Skip-gramKMeans+L2 | -0.066 |
| Word2Vec Skip-gram[Agglo +L2] | -0.066 |
| Word2Vec Skip-gram | -0.092 |

### WordSim Spearman Correlation

| Dataset | Model | Spearman ρ |
|---------|-------|------------|
| WordSim353 | Word2Vec CBOW | 0.181 |
| WordSim353_JA-EN | Word2Vec CBOW | 0.181 |
| WordSim353 | Word2Vec Skip-gram | 0.333 |
| WordSim353_JA-EN | Word2Vec Skip-gram | 0.333 |
| WordSim353 | BERT (Base) | 0.294 |
| WordSim353_JA-EN | BERT (Base) | 0.294 |
| WordSim353 | BERT (Fine-Tuned) | 0.392 |
| WordSim353_JA-EN | BERT (Fine-Tuned) | 0.392 |

## Figures

Figures for this run are in the `figures/` directory corresponding to seed 42.
Note: Negative ARI values indicate agreement lower than chance (not an error).

