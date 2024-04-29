# Running the ablation study

1. Make sure the appropriate jsonl files are present in `results` directory
2. Download the `parquet` files for each of the datasets as described below

### OBQA
```
wget -O obqa.parquet "https://huggingface.co/datasets/allenai/openbookqa/resolve/main/main/test-00000-of-00001.parquet?download=true"
```

### Quartz
```
wget https://huggingface.co/datasets/allenai/quartz/resolve/main/data/test-00000-of-00001.parquet?download=true -o quartz.parquet
```

### TruthfulQA
```
wget -O truthfulqa.parquet "https://huggingface.co/datasets/truthful_qa/resolve/main/multiple_choice/validation-00000-of-00001.parquet?download=true"
```