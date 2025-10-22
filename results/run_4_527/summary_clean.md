# Paper 1 Pipeline Summary: CLEAN Mode

- **Run Seed**: 527
- **Corpus Path**: `txt_corpus\W2V_Corpus_CL.txt`
- **Corpus SHA-256**: `6edceac27f560cd4a869846f28a5371b40775beb84b6ca28e1307c82791f03fb`

## Model Hashes

- **Word2Vec CBOW SHA-256**: `e8e6c5594864c355aabb427ec681ea54d8ceac04e9ffc65cb6c486aad60f943e`
- **Word2Vec Skip-gram SHA-256**: `53588bf0e7b3b9091f45ba21979ec4e7585ef4721668c090d3441ccea8b5b0d5`
- **Fine-Tuned BERT (dir hash)**: `f22f253bcede44cca5b6e02eb9958870c834919e098bdd8e166969892a016df1`

### ARI Scores: physical_vs_conceptual

| Model | ARI Score |
|-------|-----------|
| BERT Cased (Final) | 0.718 |
| BERT Cased (Final)KMeans+L2 | 0.718 |
| BERT Cased (Final)[Agglo +L2] | 0.718 |
| Word2Vec Skip-gramKMeans+L2 | 0.208 |
| Word2Vec CBOW | 0.133 |
| Word2Vec Skip-gram[Agglo +L2] | 0.036 |
| Word2Vec CBOWKMeans+L2 | -0.100 |
| Word2Vec CBOW[Agglo +L2] | -0.100 |
| Word2Vec Skip-gram | -0.109 |

### ARI Scores: tokonoma_as_conceptual

| Model | ARI Score |
|-------|-----------|
| BERT Cased (Final) | 0.597 |
| BERT Cased (Final)KMeans+L2 | 0.597 |
| BERT Cased (Final)[Agglo +L2] | 0.597 |
| Word2Vec Skip-gramKMeans+L2 | 0.138 |
| Word2Vec CBOW | 0.100 |
| Word2Vec Skip-gram[Agglo +L2] | 0.001 |
| Word2Vec CBOWKMeans+L2 | -0.078 |
| Word2Vec CBOW[Agglo +L2] | -0.078 |
| Word2Vec Skip-gram | -0.092 |

### WordSim Spearman Correlation

| Dataset | Model | Spearman ρ |
|---------|-------|------------|
| WordSim353 | Word2Vec CBOW | 0.177 |
| WordSim353_JA-EN | Word2Vec CBOW | 0.177 |
| WordSim353 | Word2Vec Skip-gram | 0.325 |
| WordSim353_JA-EN | Word2Vec Skip-gram | 0.325 |
| WordSim353 | BERT (Base) | 0.294 |
| WordSim353_JA-EN | BERT (Base) | 0.294 |
| WordSim353 | BERT (Fine-Tuned) | 0.375 |
| WordSim353_JA-EN | BERT (Fine-Tuned) | 0.375 |

## Figures

Figures for this run are in the `figures/` directory corresponding to seed 527.
Note: Negative ARI values indicate agreement lower than chance (not an error).

