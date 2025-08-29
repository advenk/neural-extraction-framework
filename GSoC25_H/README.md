# Neural Hindi Wiki Triple Extraction Pipeline

This project aims to enhance and evaluate the relation extraction pipeline for Hindi Wikipedia articles, utilizing a combination of state-of-the-art language models and rule-based methods. 
This project builds on the work done in GSoC24 for the Hindi chapter. The goals of this project were multi-fold:

Weekly Blog: https://advenk.github.io/av-blog/
Hindi SPARQL temporary endpoint deployed at http://95.217.58.54:8890/sparql
Unmerged PR for Hindi mappings and extractors: https://github.com/dbpedia/extraction-framework/pull/776 - Can be merged only after updating the mappings via UI. 

### What Work was Done
* Streamline the exisiting pipeline and make it easy to run
* Implement and evaluate new triplet extraction methods using Small Language Models (SLM)
* Enhance the IndIE method by integrating SLMs in the last stage of IndIE
* Implement link prediction and evaluate to predict missing links using the existing hindi KG
* Implement predicate linking to dbpedia ontology in the existing framework
* Deploy the SPARQL endpoint and test its performance


## Progress Overview

### Week 1-2

- Streamlined the exisiting pipeline from GSoC'24, made it simple to run (steps below)
- Rebased and raised final PR for addings Hindi mappings and extractors - https://github.com/dbpedia/extraction-framework/pull/776

### Week 3 

- Simple framework for evaluating the performance of SLMs w.r.t various prompt strategies on the Hindi BenchIE dataset
- Lives in **llm_IE/** directory
- Read https://advenk.github.io/av-blog/posts/gsoc-25/gsoc-week-3/

### Week 4 

- Used ReAct framework for prompting
- Link prediction setup

### Week 5
- Link Prediction notebook completed
- Integrated gemma into the last stage of the IndIE pipeline

### Week 6
- Experimented and evaluated IndIE + gemma pipeline from previous week
- Achieved ~65% recall
- Designed finetuning plan

### Week 7
- Wrote the synthetic data generation script
- Deployed and performance tested the SPARQL endpoint (lives in https://github.com/advenk/virtuoso-sparql-endpoint-quickstart/tree/gsoc25_hindi_chapter)

### Week 8-9
- Paper Review of "Text-to-SPARQL Goes Beyond English: Multilingual Question Answering Over Knowledge Graphs through Human-Inspired Reasoning"
- Document the SPARQL endpoint (temporarily here: http://95.217.58.54:8890/sparql)
- Synthetic Data Generation script change

### Week 10
- Evaluated code for entity linking in depth
- Identified Gaps
- EL setup

### Week 11
- Implemented predicate linking to dbpedia ontology
- Integrated into existing pipeline

### Week 12
- EL normalisation to dbpedia resources from wikidata ID
- Tranlsated and scored lexical similarity for predicates
- Clean up and wrap up of code
- Consolidating and documenting

## How to Run
- Install dependencies by running in GSoC25/ root 
```pip install -r requirements.txt```

### Streamlit Demo
Run
```streamlit run src/demo.py```

### SLM guided IE
- Lives in llm_IE/
- Refer llm_IE/README.md

### Link Prediction
- Lives in link_prediction/
- 2 notebooks - one for analysis, one for actual implementation of the models

### Predicate Linking (Hindi → DBpedia Ontology)
#### Implementation strategy
We map Hindi predicate phrases to English DBpedia ontology properties (`dbo:*`) using a hybrid approach:

- Graph evidence: direct `dbo:*` edges between `dbr:S` and `dbr:O` (both directions)
- Lexical match: SPARQL label search (Hindi/English)
- Type compatibility: `rdfs:domain`/`rdfs:range` vs `rdf:type(S)`/`rdf:type(O)`
- Multilingual sentence embeddings: local index over the ontology built from `GSoC24_H/ontology_input/ontology--DEV_type=parsed.ttl`

Implementation lives in `src/predicate_linking.py`.

Scoring (higher is better): 
```python
w_graph, w_emb, w_lex, w_type = 0.4, 0.3, 0.2, 0.1
# weighted sum -> prioritize graph, then embedding, then lexical, then type
score = w_graph * graph_score + w_emb * emb_score + w_lex * lex_score + w_type * type_score
```

Outputs include `property_uri`, labels, score, direction (`S->O` or `O->S`), and evidence breakdown.

#### One-time setup
Embeddings:
Automated creation of ontology property and embeddings files upon first run.
Cached artifacts written under `GSoC24_H/models/ontology/`:
- `dbpedia_property_catalog.json`
- `dbpedia_property_embeddings.npy`
- `dbpedia_property_index.json`

To reset caches, delete the files above and rerun.

#### How to run

- CLI (coref → RE → EL → predicate linking):
```bash
python GSoC24_H/src/start.py \
  --input_dir GSoC24_H/input \
  --do_coref --do_rel --do_el --do_prop_link --verbose
```
Predicate linking logs progress phase-wise (graph, lexical, embedding, typing, scoring) and prints top candidate.

- Streamlit demo:
```bash
streamlit run GSoC24_H/src/demo.py
```
Enable Relation Extraction, optionally Entity Linking, then Predicate Linking to view property URIs and scores.



## Limitations and Future Work

- Predicate linking currently only handles relational linking but not type linking. Type linking would require us to identify if a triplet is of the form (subject, "है", object) and then link it as the subject -> type -> object. For other triplets (relational), this is handled. The major difference is the "object" in a type linking is a class in the ontology whereas in regular relational linking this is a property. 
- Refine the pipeline for enhanced accuracy.
- Extend the pipeline to support additional languages.
