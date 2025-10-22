# Paper 1 Pipeline Summary: FULL Mode

- **Run Seed**: 777
- **Corpus Path**: `txt_corpus\W2V_Corpus.txt`
- **Corpus SHA-256**: `0aa89f31a3eb7fa49cdd875256e2ef18c70c57d99262263df661948bfd63212a`

## Model Hashes

- **Word2Vec CBOW SHA-256**: `8e341f829233a8756032167bc8409ef076b7288c803088a17bbd1b83d60a33f9`
- **Word2Vec Skip-gram SHA-256**: `e592e9e31aaa11dff2371854e2b82904c8952258c7719e38e2fadb9dcaeaf6fd`
- **Fine-Tuned BERT (dir hash)**: `e2d761d8eb990a69d5b2e48272259b98d866db005814aa5c34f78d9653b0ceb2`

### ARI Scores: physical_vs_conceptual

| Model | ARI Score |
|-------|-----------|
| BERT Cased (Final) | 0.852 |
| BERT Cased (Final)KMeans+L2 | 0.852 |
| Word2Vec Skip-gram[Agglo +L2] | 0.476 |
| BERT Cased (Final)[Agglo +L2] | 0.394 |
| Word2Vec Skip-gramKMeans+L2 | 0.376 |
| Word2Vec CBOW | 0.133 |
| Word2Vec CBOWKMeans+L2 | -0.019 |
| Word2Vec CBOW[Agglo +L2] | -0.081 |
| Word2Vec Skip-gram | -0.109 |

### ARI Scores: tokonoma_as_conceptual

| Model | ARI Score |
|-------|-----------|
| BERT Cased (Final) | 0.719 |
| BERT Cased (Final)KMeans+L2 | 0.719 |
| Word2Vec Skip-gram[Agglo +L2] | 0.373 |
| BERT Cased (Final)[Agglo +L2] | 0.304 |
| Word2Vec Skip-gramKMeans+L2 | 0.285 |
| Word2Vec CBOW | 0.100 |
| Word2Vec CBOWKMeans+L2 | -0.036 |
| Word2Vec CBOW[Agglo +L2] | -0.066 |
| Word2Vec Skip-gram | -0.092 |

### WordSim Spearman Correlation

| Dataset | Model | Spearman ρ |
|---------|-------|------------|
| WordSim353 | Word2Vec CBOW | 0.189 |
| WordSim353_JA-EN | Word2Vec CBOW | 0.189 |
| WordSim353 | Word2Vec Skip-gram | 0.334 |
| WordSim353_JA-EN | Word2Vec Skip-gram | 0.334 |
| WordSim353 | BERT (Base) | 0.294 |
| WordSim353_JA-EN | BERT (Base) | 0.294 |
| WordSim353 | BERT (Fine-Tuned) | 0.343 |
| WordSim353_JA-EN | BERT (Fine-Tuned) | 0.343 |

## Figures

Figures for this run are in the `figures/` directory corresponding to seed 777.
Note: Negative ARI values indicate agreement lower than chance (not an error).

