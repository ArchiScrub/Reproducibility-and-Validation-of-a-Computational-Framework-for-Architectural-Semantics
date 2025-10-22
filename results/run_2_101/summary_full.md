# Paper 1 Pipeline Summary: FULL Mode

- **Run Seed**: 101
- **Corpus Path**: `txt_corpus\W2V_Corpus.txt`
- **Corpus SHA-256**: `0aa89f31a3eb7fa49cdd875256e2ef18c70c57d99262263df661948bfd63212a`

## Model Hashes

- **Word2Vec CBOW SHA-256**: `93619f77da7a3b17cdb05a55fba70fbc86895eb70e23d59a2a5e872fbd8665d3`
- **Word2Vec Skip-gram SHA-256**: `6c3774b1fb4bbf9b508ea155d7abd8732c89ccf8e48da4e459b60ad937c31f90`
- **Fine-Tuned BERT (dir hash)**: `1dd110b96e685a4a34bf40762da2837863ac12be7707426b53c852a408b20f71`

### ARI Scores: physical_vs_conceptual

| Model | ARI Score |
|-------|-----------|
| BERT Cased (Final) | 0.852 |
| BERT Cased (Final)KMeans+L2 | 0.852 |
| Word2Vec Skip-gram[Agglo +L2] | 0.587 |
| BERT Cased (Final)[Agglo +L2] | 0.394 |
| Word2Vec CBOW | 0.133 |
| Word2Vec CBOWKMeans+L2 | -0.025 |
| Word2Vec CBOW[Agglo +L2] | -0.081 |
| Word2Vec Skip-gramKMeans+L2 | -0.100 |
| Word2Vec Skip-gram | -0.110 |

### ARI Scores: tokonoma_as_conceptual

| Model | ARI Score |
|-------|-----------|
| BERT Cased (Final) | 0.719 |
| BERT Cased (Final)KMeans+L2 | 0.719 |
| Word2Vec Skip-gram[Agglo +L2] | 0.474 |
| BERT Cased (Final)[Agglo +L2] | 0.304 |
| Word2Vec CBOW | 0.100 |
| Word2Vec CBOWKMeans+L2 | -0.002 |
| Word2Vec CBOW[Agglo +L2] | -0.054 |
| Word2Vec Skip-gramKMeans+L2 | -0.066 |
| Word2Vec Skip-gram | -0.095 |

### WordSim Spearman Correlation

| Dataset | Model | Spearman ρ |
|---------|-------|------------|
| WordSim353 | Word2Vec CBOW | 0.158 |
| WordSim353_JA-EN | Word2Vec CBOW | 0.158 |
| WordSim353 | Word2Vec Skip-gram | 0.312 |
| WordSim353_JA-EN | Word2Vec Skip-gram | 0.312 |
| WordSim353 | BERT (Base) | 0.294 |
| WordSim353_JA-EN | BERT (Base) | 0.294 |
| WordSim353 | BERT (Fine-Tuned) | 0.325 |
| WordSim353_JA-EN | BERT (Fine-Tuned) | 0.325 |

## Figures

Figures for this run are in the `figures/` directory corresponding to seed 101.
Note: Negative ARI values indicate agreement lower than chance (not an error).

