# Paper 1 Pipeline Summary: FULL Mode

- **Run Seed**: 387
- **Corpus Path**: `txt_corpus\W2V_Corpus.txt`
- **Corpus SHA-256**: `0aa89f31a3eb7fa49cdd875256e2ef18c70c57d99262263df661948bfd63212a`

## Model Hashes

- **Word2Vec CBOW SHA-256**: `b20c7874d0069dca43a905e4620244e2358ba90251683d249430414ebe764f73`
- **Word2Vec Skip-gram SHA-256**: `b2ac786f9269aa388ed4d61763c307c64a0c66aabcdf3645df45655a074f9e55`
- **Fine-Tuned BERT (dir hash)**: `68245c7375275073cb40746d6e86032eaa5d788ebc0026a32bf6f39c8b64638b`

### ARI Scores: physical_vs_conceptual

| Model | ARI Score |
|-------|-----------|
| BERT Cased (Final) | 0.852 |
| BERT Cased (Final)KMeans+L2 | 0.852 |
| Word2Vec Skip-gram[Agglo +L2] | 0.587 |
| Word2Vec Skip-gramKMeans+L2 | 0.476 |
| BERT Cased (Final)[Agglo +L2] | 0.394 |
| Word2Vec CBOWKMeans+L2 | 0.208 |
| Word2Vec CBOW[Agglo +L2] | 0.208 |
| Word2Vec CBOW | 0.133 |
| Word2Vec Skip-gram | -0.054 |

### ARI Scores: tokonoma_as_conceptual

| Model | ARI Score |
|-------|-----------|
| BERT Cased (Final) | 0.719 |
| BERT Cased (Final)KMeans+L2 | 0.719 |
| Word2Vec Skip-gram[Agglo +L2] | 0.474 |
| Word2Vec Skip-gramKMeans+L2 | 0.373 |
| BERT Cased (Final)[Agglo +L2] | 0.304 |
| Word2Vec CBOWKMeans+L2 | 0.138 |
| Word2Vec CBOW[Agglo +L2] | 0.138 |
| Word2Vec CBOW | 0.100 |
| Word2Vec Skip-gram | -0.021 |

### WordSim Spearman Correlation

| Dataset | Model | Spearman ρ |
|---------|-------|------------|
| WordSim353 | Word2Vec CBOW | 0.179 |
| WordSim353_JA-EN | Word2Vec CBOW | 0.179 |
| WordSim353 | Word2Vec Skip-gram | 0.343 |
| WordSim353_JA-EN | Word2Vec Skip-gram | 0.343 |
| WordSim353 | BERT (Base) | 0.294 |
| WordSim353_JA-EN | BERT (Base) | 0.294 |
| WordSim353 | BERT (Fine-Tuned) | 0.345 |
| WordSim353_JA-EN | BERT (Fine-Tuned) | 0.345 |

## Figures

Figures for this run are in the `figures/` directory corresponding to seed 387.
Note: Negative ARI values indicate agreement lower than chance (not an error).

