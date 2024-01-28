
# RTS-CESP

This is the repository for the manuscript **“Chemoenzymatic Synthesis Planning Guided by Reaction Type Score”**.

## Reaction Type score (RTscore)

The Reaction Type score (RTscore) distinguishes synthesis reactions from decomposition reactions, and we trained two RTscore models using the USPTO and ECREACT databases separately for chemical and biological reactions.

### Dependencies

- RDKit
- PyTorch
- NumPy

### Model
 The standalone model for chemical reactions is defined at:  
  `RTscore/RTscore_chem/RTscore_chem_API.py`

 The standalone model for biological reactions is defined at:  
  `RTscore/RTscore_bio/RTscore_bio_API.py`
  
### Training
To train with your own database, use the code available in the `data_processing` directory.

## Chemoenzymatic synthesis planning using ML-based methods

Install the package:

```shell
pip install rxn4chemistry
```

Run your target molecules through the following script:

```shell
model_eval/10_examples/runmodel_IBM.py
```

For additional information about rxn4chemistry, see [rxn4chemistry Github](https://github.com/rxn4chemistry/rxn4chemistry).

The 10 examples to evaluate RTscore are available at:  
`model_eval/10_examples/results`


## Chemoenzymatic synthesis planning using rule-based methods

1. Create a Conda environment and download Aizythfinder checkpoints according to [Aizythfinder Github](https://github.com/MolecularAI/aizynthfinder).

2. Clone Aizythfinder repository:

   ```shell
   git clone https://github.com/MolecularAI/aizynthfinder.git model_eval/1000_molecules
   ```

3. Run your target molecules using the following script:

   ```shell
   model_eval/1000_molecules/runmodel_aiz_retrobiocat.py
   ```

The validation results on 1000 molecules are available at:  
`model_eval/1000_molecules/1000molecules_results.json`


