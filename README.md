# dnnmicrobiome

- Deep Neural network on microbiome from start to finish for label data
- Your unknown sequences should be a CSV file with no label.
- Fastai approach

```
python dna_dnn_cli.py train --input microbiome.csv --epochs 10 --lr 5e-4 --model-out model.pt
python dna_dnn_cli.py predict --input new_seqs.csv --model-in model.pt --preds-out out.csv

```

Gaurav Sablok \
gsablok@proton.me