# PRSummarizer

The source code of the paper "Automatic Generation of Pull Request Description".

## Dataset
### Raw Data

Our collected 333K pull requests can be downloaded from [here](https://drive.google.com/drive/folders/1VMByXOEmJDQL_JQY6l63NRiveUySY-Sq?usp=sharing). Here is a PR example in the json file:

```json
{
    "id": "elastic/elasticsearch_37980",
    "body": "'Eclipse build files were missing so .eclipse project files were not being generated.\\r\\nCloses #37973\\r\\n\\r\\n'",
    "cms": [
      "'Added missing eclipse-build.gradle files\\n\\nCloses #fix/37973'"
    ],
    "commits": {
      "'3e10ee798c932cc1cab1ea6ca679417408fc1416'": {
        "cm": "'Added missing eclipse-build.gradle files\\n\\nCloses #fix/37973'",
        "comments": []
      }
    }
  }
```

- id: `$user/$project_$prid`
- body: PR description
- cms: the commit messages in this PR
- commis: the commits in this PR
    - key is the SHA1 hash digest
        - cm: commit message
        - comments: source code comments added in this commit

### Preprocessed Dataset

Our dataset can be downloaded from [here](https://drive.google.com/drive/folders/1VMByXOEmJDQL_JQY6l63NRiveUySY-Sq?usp=sharing), which contains:

- the train, validation and test sets
- a json file for building vocabulary

### Regular Expressions

To preprocess the raw data, we used the following regular expressions:

```python
email_pattern = r'(^|\s)<[\w.-]+@(?=[a-z\d][^.]*\.)[a-z\d.-]*[^.]>'
url_pattern = r'https?://[-a-zA-Z0-9@:%._+~#?=/]+(?=($|[^-a-zA-Z0-9@:%._+~#?=/]))'
reference_pattern = r'#[\d]+'
signature_pattern = r'^(signed-off-by|co-authored-by|also-by):'
at_pattern = r'@\S+'
structure_pattern = r'^#+'
version_pattern = r'(^|\s|-)[\d]+(\.[\d]+){1,}'
sha_pattern = r'(^|\s)[\dA-Fa-f-]{7,}(?=(\s|$))'
digit_pattern = r'(^|\s|-)[\d]+(?=(\s|$))'
```

## Installation
### Clone and Prepare Dataset
```bash
$ git clone https://github.com/Tbabm/PRSummarizer.git
$ cd PRSummarizer
$ mkdir data
# download our preprocessed dataset and place the four files in `data`
$ mkdir models
```

### Install ROUGE
- See [here](https://gist.github.com/Tbabm/65b5d8a3adb9845d55ce27143913e3b2) for instructions about installing ROUGE
- Please make sure you have correctly set environment variable `ROUGE` to `/absolute/path/to/ROUGE-RELEASE-1.5.5`

### Install Dependencies
**Through conda**:

```bash
$ conda env create -f environment.yml
```

**OR** through pip

```bash
$ pip install -r requirements.txt
```

### Install pyrouge
- install and test pyrouge if you haven't done it.

```bash
$ git clone https://github.com/bheinzerling/pyrouge
$ cd pyrouge
$ pip install .

# set rouge path for pyrouge
$ pyrouge_set_rouge_path ${ROUGE}

# test the installation of pyrouge
$ python -m pyrouge.test
```

## Usage 
### Train

Train `Attn+PG` first:

```bash
python3 -m prsum.prsum train --param-path params_attn_pg.json
```

After training, suppose the models are stored in `models/train_12345678/model/`. Select the best `Attn+PG` model:

```bash
python3 -m prsum.prsum select_model \
                       --param_path params_attn_pg.json \
                       --model_pattern "models/train_12345678/model/model_{}_" \
                       --start_iter 1000 \
                       --end_iter 26000
```

Suppose the best model is `model_12000_87654321`. Train `Attn+PG+RL` based on the best model:

```bash
python3 -m prsum.prsum train \
                       --param_path params_attn_pg_rl.json \
                       --model_path "models/train_12345678/model/model_12000_87654321"
```

### Validate

Select the best `Attn+PG` model:

```bash
# start_iter = the best iteration of `Attn+PG` (here, 12000) + save_interval (here, 1000)
START_ITER=13000
python3 -m prsum.prsum select_model
                       --param_path params_attn_pg_rl.json \
                       --model_pattern "models/train_12345678/model/rl_model_{}_" \
                       --start_iter $START_ITER \
                       --end_iter 41000
```

Suppose the best model is `model_34000_98765432`.

### Test

Test the best `Attn+PG+RL` model:

```bash
python3 -m prsum.prsum decode \
                       --param_path params_attn_pg_rl.json \
                       --model_path "models/train_12345678/model/rl_model_34000_98765432" \
                       --ngram_filter 1
```

Now, you will get the test results.

NOTE: Your test results may be slightly different from those reported in our paper. Because the pointer generator uses the `scatter_add` function in pytorch. When using GPUs, this function is undeterministic. See [here](https://pytorch.org/docs/stable/notes/randomness.html) for more details.

### Pre-trained Model

Our pre-trained model and test results can be downloaded [here](https://drive.google.com/drive/folders/1VMByXOEmJDQL_JQY6l63NRiveUySY-Sq?usp=sharing). To test with our pre-atrained model:

```bash
mkdir models
mv rl_model_34000 ./models
python3 -m prsum.prsum decode \
                       --param_path params_attn_pg_rl.json \
                       --model_path "models/rl_model_34000" \
                       --ngram_filter 1
``` 

## Reference

- Our paper: "Automatic Generation of Pull Request Description"
- https://github.com/atulkum/pointer_summarizer
- https://github.com/rohithreddy024/Text-Summarizer-Pytorch
