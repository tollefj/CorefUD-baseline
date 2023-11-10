## A simplified multilingual coreference baseline model for CorefUD
Based on https://github.com/ufal/crac2022-corpipe

# Setup

1. Fetch data.
    ```bash
    cd data
    chmod +x get.sh
    ./get.sh
    ```
    - If you wish to reduce the number of languages, either edit the `get.sh` file accordingly, or delete/move folders.
2. Convert all data to jsonlines (if needed)
    ```bash
    cd data_handling
    chmod +x corefud_convert.sh
    ./corefud_convert.sh
    ```
3. Train the model in `src/models/simple-corpipe`. Example:
    ```bash
    cd src/models/simple-corpipe
    python3 train.py --langs germanic
    ```
    This will run the "germanic languges" found in the following definitions:
    ```
    romance_langs = "ca_ancora es_ancora fr_democrat".split()
    germanic_langs = "de_parcorfull de_potsdamcc en_gum en_parcorfull no_bokmaalnarc no_nynorsknarc".split()
    slavic_baltic_langs = "cs_pcedt cs_pdt pl_pcc lt_lcc ru_rucor".split()
    urgic_turkic_langs = "hu_korkor hu_szegedkoref tr_itcc".split()

    langs_dict = {
        "romance": romance_langs,
        "germanic": germanic_langs,
        "slavic": slavic_baltic_langs,
        "urgic": urgic_turkic_langs,
        "all": langs,
    }
    ```
    Omitting any args will default to "all", which requires all languages in the `data` folder.

## Training configuration:

| Argument              | Default           | Type       | Description                                      |
|-----------------------|-------------------|------------|--------------------------------------------------|
| --langs               | []                | List[str]  | Languages to train on.                            |
| --batch_size          | 16                | int        | Batch size.                                      |
| --bert                | xlm-roberta-base  | str        | Bert model.                                      |
| --debug               | False             | bool       | Debug mode.                                      |
| --epochs              | 10                | int        | Number of epochs.                                |
| --exp                 | run               | str        | Exp name.                                        |
| --label_smoothing     | 0.0               | float      | Label smoothing.                                 |
| --learning_rate       | 2e-5              | float      | Learning rate.                                   |
| --learning_rate_decay | False             | bool       | Decay LR.                                        |
| --max_links           | None              | int        | Max antecedent links to train on.                |
| --right               | 50                | int        | Reserved space for right context, if any.        |
| --seed                | 42                | int        | Random seed.                                     |
| --segment             | 512               | int        | Segment size.                                    |
| --train               | []                | List[str]  | Additional train data.                           |
| --warmup              | 0.1               | float      | Warmup ratio.                                    |