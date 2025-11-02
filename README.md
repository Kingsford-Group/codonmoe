# CodonMoE


CodonMoE is a Python package that implements Adaptive Mixture of Codon Reformative Experts (CodonMoE)  for RNA analyses.

## Datasets

We include four public mRNA datasets, all bundled as CSVs in `datasets/`. Each file shares the same schema:
- `Sequence`: RNA sequence (A,U,C,G)
- `Value`: real-valued target
- `Dataset`: dataset identifier
- `Split`: `train` / `valid` / `test`

| Dataset | File |
| --- | --- |
| MLOS | `datasets/MLOS.csv` |
| Tc-Riboswitches | `datasets/Tc-Riboswitches.csv` |
| mRFP Expression | `datasets/mRFP_Expression.csv` |
| CoV Vaccine Degradation | `datasets/CoV_Vaccine_Degradation.csv` |



## Installation

You can install CodonMoE using below command:

```bash
# pip install codonmoe
python setup.py install
```

## API Reference

### CodonMoE

```python
CodonMoE(input_dim, num_experts=4, dropout_rate=0.1)
```

Parameters:
- `input_dim`: Dimension of the input features
- `num_experts`: Number of expert networks in the Mixture of Experts
- `dropout_rate`: Dropout rate for regularization

### mRNAModel

```python
mRNAModel(base_model, codon_moe)
```

Parameters:
- `base_model`: The base model to be integrated with CodonMoE
- `codon_moe`: The CodonMixture of Experts model

## API Tests

```bash
python -m unittest tests/test_codon_moe.py
```


## Citation

If you find this repository useful, please cite our paper: [CodonMoE: DNA Language Models for mRNA Analyses](https://arxiv.org/abs/2508.04739).

```bibtex
@article{du2025codonmoe,
  title={CodonMoE: DNA Language Models for mRNA Analyses},
  author={Du, Shiyi and Liang, Litian and Li, Jiayi and Kingsford, Carl},
  journal={arXiv preprint arXiv:2508.04739},
  year={2025}
}
```

