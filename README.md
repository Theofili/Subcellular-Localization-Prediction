

# Subcellular Localization Prediction Using Pre-trained Mistral Models

This markdown contains the full workflow of creating a function which predicts the location of user inputed proteins.

# 1. Collecting Data

## In the `data/data_functions` folder run:

### 1.1. `average.py`
When executed this program:
* Collects **human protein sequences** from the Entrez-protein database. 
* Calculates the lengths of the sequences.
* Prints statistics like mean, sd and distribution of lengths.
* Prints **lower (5%) and upper (95%) bound** for sequence lengths.

These numbers are used to filter out extremely short or long protein sequences, because they are often biological artifacts (fragments, chimeras, or annotation errors) and don’t represent typical proteins. From a computational side, very short or long sequences create inefficiencies in batching, increase memory cost, and can destabilize training. By restricting lengths to a reasonable range, the model focuses on biologically meaningful proteins and trains more efficiently.

*Average length is calculated on proteins from human and not only from the ones used to train - if trained data is used upper bound is higher due to in general longer extracellular protein sequences. Don't know which bounds should be used.*

### 1.2 `model_dataframes.py`
When executed this program:
* Creates dataframes with proteins pulled from Entrez which have a *particular word* in title. In this case it is cell locations(nuclear, membrane, mitochondrial, etc.)
* Filters through duplicates and sequence lengths based on the average length.
* **Creates 50:50 binary classification dataframes for each cell location** -- Adjust sample sizes to number of target sequences
* Dataframes saved in `data/model_data` folder (naming = `model_{cell_location}_data.csv`)

Each cell location contains different numbers of sequences, and for that reason total number of sequences differ. Each dataframe contains a `sequence` column and a `type` column. When `type==1` sequence exist in the location written in the title of the dataframe, when `type==0` sequence exists in any other location in the cell.

***Doesn't use** user input, collects data from 9 cellular locations (Reticulum, Extracellular, Golgi, Membrane, Mitochondria, Nuclear, Peroxisome, Reticulum, Ribosome)*

---

All dataframes created are saved as csv's, as well as saved fasta files from Entrez.


# 2. Fine tuning the models

## In the `models folder` run:

### 2.1 `Model_{cellular_location}.py`

When executed these programms:
* Initializes training parameters.
* Splits coresponding dataframe into training(80%), validation(10%), testing(10%) sets. *Saved as csv's*
* Trains the `RaphaelMourad/Mistral-Peptide-v1-15M` model on the datasets.
* Saves results each epoch.
* Saves the model in each own folder.

Each program needs to run individually and beacause of different training dataset sizes, some models perform better than others.

After filtering:

|Type|Sequences Count|
|-----|-------------|
|Extracellular|160|
|Golgi|456|
|Mitochondrial|2749|
|Membrane|3698|
|Nuclear|2561|
|Peroxisome|236|
|Reticulum|431|
|Ribosome|164|



**Most important parameters:**
* Learning rate: 2e-5
* Number of epochs: 20
* Max length: 1000
* Early stopping patience: 3

Some other paramters can be changed for the model to train faster, in better performing hardware (batch_size, number of epochs, bf16)

*Lysosome dataframe did not have sufficient amount of sequences, therefore there is no model.*

### 2.2 `Localization.py`
When executed this program:
* Puts a user-inputted sequence through all models.
* **Prints the model that yields the highest score**
* Prints all model scores
* If no model hits a score above 0.5, `'Not Matched'` is printed
