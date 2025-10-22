# Paper 1 Pipeline Summary: CLEAN Mode

- **Run Seed**: 387
- **Corpus Path**: `txt_corpus\W2V_Corpus_CL.txt`
- **Corpus SHA-256**: `6edceac27f560cd4a869846f28a5371b40775beb84b6ca28e1307c82791f03fb`

## Model Hashes

- **Word2Vec CBOW SHA-256**: `2df8b10c2cdb0019df1657adb5cecec458ea67588ca334bb4700da72831cb6e0`
- **Word2Vec Skip-gram SHA-256**: `8d58e3022e542d13164a85a23397e329f30269fd61924d92eb0abd7bb4300c28`
- **Fine-Tuned BERT (dir hash)**: `b48fe380f60ee11b20f999e66c84eb616b00eba01c8d5744fea57fff8bc13149`

### ARI Scores: physical_vs_conceptual

| Model | ARI Score |
|-------|-----------|
| Word2Vec Skip-gram[Agglo +L2] | 0.848 |
| BERT Cased (Final) | 0.718 |
| BERT Cased (Final)KMeans+L2 | 0.718 |
| BERT Cased (Final)[Agglo +L2] | 0.718 |
| Word2Vec Skip-gramKMeans+L2 | 0.376 |
| Word2Vec CBOW | 0.133 |
| Word2Vec CBOWKMeans+L2 | -0.002 |
| Word2Vec CBOW[Agglo +L2] | -0.100 |
| Word2Vec Skip-gram | -0.110 |

### ARI Scores: tokonoma_as_conceptual

| Model | ARI Score |
|-------|-----------|
| Word2Vec Skip-gram[Agglo +L2] | 0.710 |
| BERT Cased (Final) | 0.597 |
| BERT Cased (Final)KMeans+L2 | 0.597 |
| BERT Cased (Final)[Agglo +L2] | 0.597 |
| Word2Vec Skip-gramKMeans+L2 | 0.285 |
| Word2Vec CBOW | 0.100 |
| Word2Vec CBOWKMeans+L2 | 0.032 |
| Word2Vec CBOW[Agglo +L2] | -0.078 |
| Word2Vec Skip-gram | -0.095 |

### WordSim Spearman Correlation

| Dataset | Model | Spearman ρ |
|---------|-------|------------|
| WordSim353 | Word2Vec CBOW | 0.168 |
| WordSim353_JA-EN | Word2Vec CBOW | 0.168 |
| WordSim353 | Word2Vec Skip-gram | 0.333 |
| WordSim353_JA-EN | Word2Vec Skip-gram | 0.333 |
| WordSim353 | BERT (Base) | 0.294 |
| WordSim353_JA-EN | BERT (Base) | 0.294 |
| WordSim353 | BERT (Fine-Tuned) | 0.317 |
| WordSim353_JA-EN | BERT (Fine-Tuned) | 0.317 |

## Figures

Figures for this run are in the `figures/` directory corresponding to seed 387.
Note: Negative ARI values indicate agreement lower than chance (not an error).

