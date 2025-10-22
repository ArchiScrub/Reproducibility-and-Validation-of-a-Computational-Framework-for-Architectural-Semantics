# Paper 1 Pipeline Summary: CLEAN Mode

- **Run Seed**: 1234
- **Corpus Path**: `txt_corpus\W2V_Corpus_CL.txt`
- **Corpus SHA-256**: `6edceac27f560cd4a869846f28a5371b40775beb84b6ca28e1307c82791f03fb`

## Model Hashes

- **Word2Vec CBOW SHA-256**: `33affc79fb99a53f839be999471d63e52d84bbd7b6a6f6e8f3276dc8510428a9`
- **Word2Vec Skip-gram SHA-256**: `5e15fbe1d0c455a23dfc6434f82f3465663fb2223e58827b848dc90998d1bc80`
- **Fine-Tuned BERT (dir hash)**: `7729b4ce54a0899d642c221d3d8cdefe427c3ecb75f7a095d2526286e133c4e3`

### ARI Scores: physical_vs_conceptual

| Model | ARI Score |
|-------|-----------|
| BERT Cased (Final) | 0.718 |
| BERT Cased (Final)KMeans+L2 | 0.718 |
| BERT Cased (Final)[Agglo +L2] | 0.718 |
| Word2Vec Skip-gramKMeans+L2 | 0.711 |
| Word2Vec CBOW | 0.133 |
| Word2Vec CBOWKMeans+L2 | -0.029 |
| Word2Vec Skip-gram | -0.100 |
| Word2Vec Skip-gram[Agglo +L2] | -0.100 |
| Word2Vec CBOW[Agglo +L2] | -0.109 |

### ARI Scores: tokonoma_as_conceptual

| Model | ARI Score |
|-------|-----------|
| BERT Cased (Final) | 0.597 |
| BERT Cased (Final)KMeans+L2 | 0.597 |
| BERT Cased (Final)[Agglo +L2] | 0.597 |
| Word2Vec Skip-gramKMeans+L2 | 0.585 |
| Word2Vec CBOW | 0.100 |
| Word2Vec CBOWKMeans+L2 | -0.021 |
| Word2Vec Skip-gram | -0.078 |
| Word2Vec Skip-gram[Agglo +L2] | -0.078 |
| Word2Vec CBOW[Agglo +L2] | -0.092 |

### WordSim Spearman Correlation

| Dataset | Model | Spearman ρ |
|---------|-------|------------|
| WordSim353 | Word2Vec CBOW | 0.174 |
| WordSim353_JA-EN | Word2Vec CBOW | 0.174 |
| WordSim353 | Word2Vec Skip-gram | 0.332 |
| WordSim353_JA-EN | Word2Vec Skip-gram | 0.332 |
| WordSim353 | BERT (Base) | 0.294 |
| WordSim353_JA-EN | BERT (Base) | 0.294 |
| WordSim353 | BERT (Fine-Tuned) | 0.321 |
| WordSim353_JA-EN | BERT (Fine-Tuned) | 0.321 |

## Figures

Figures for this run are in the `figures/` directory corresponding to seed 1234.
Note: Negative ARI values indicate agreement lower than chance (not an error).

