# Evaluation of different configurations

This evaluation folder contains all the scripts and configurations to reproduce the results of the results presented in the bachelor thesis.

### Folder Structure

```
evaluation/
├── components/
│   ├── base.py
│   ├── embeddings.py
│   ├── llms.py
│   └── retriever.py
├── registry/
│   ├── embeddings/
│   │   ├── text_embedding_ada_002.yaml
│   │   └── ...
│   ├── llm/
│   │   ├── text-davinci-003.yaml
│   │   └── ...
│   ├── model_config/
│   │   ├── baseline.yaml
│   │   └── ...
│   ├── prompt/
│   │   ├── baseline.yaml
│   │   └── ...
│   ├── retriever/
│   │   ├── csv.yaml
│   │   ├── redis.yaml
│   │   └── ...
│   └── model_graded/
│       └── ...
├── data/
│   ├── information_retriever/
│   │   └── data_example.jsonl
│   └── answer_generator/
│       └── data_example.jsonl
├── final_runs/
│   └── ...
├── eval.py
├── registry.py
└── run.py
```
