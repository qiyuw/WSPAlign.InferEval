# WSPAligner-inference
This project provides the inference libarary for [WSPAlign](https://github.com/qiyuw/WSPAlign).

## Requirements
Run `pip install -r requirements.txt` to install the required packages.

### SpaCy
We use SpaCy to tokenize sentences. For now this libaracy supports six langauges. Declare your source and target langauges with `--src_tokenizer_lang` and `--tgt_tokenizer_lang`.

| Language abbreviation | Language|
|-------|-------|
| en | English|
| ja | Japanese|
| zh | Chinese|
| fr | French |
| de | German |
| ro | Romanian|

You also need to install Spacy language package with `pip install [ja]` for languages (e.g., ja for Japanese) you want to use.

Please refer to [https://spacy.io/](https://spacy.io/) for more information. You can easily apply languages other than the above six, but note that for now we do not provide finetuned WSPAligner for other languages. WSPAligner in other languages can only perform in a zero-shot way with our pre-trained model.

## Inference
Run `python inference.py --model_name_or_path qiyuw/WSPAlign-ft-kftt --src_tokenizer_lang ja --src_text="私は猫が好きです。" --tgt_tokenizer_lang en --tgt_text="I like cats."`

### Model list
| Model List| Description|
|-------|-------|
|[qiyuw/WSPAlign-xlm-base](https://huggingface.co/qiyuw/WSPAlign-xlm-base) | Pretrained on xlm-roberta |
|[qiyuw/WSPAlign-mbert-base](https://huggingface.co/qiyuw/WSPAlign-mbert-base) | Pretrained on mBERT|
|[qiyuw/WSPAlign-ft-kftt](https://huggingface.co/qiyuw/WSPAlign-ft-kftt)| Finetuned with English-Japanese KFTT dataset|
|[qiyuw/WSPAlign-ft-deen](https://huggingface.co/qiyuw/WSPAlign-ft-deen)| Finetuned with German-English dataset|
[qiyuw/WSPAlign-ft-enfr](https://huggingface.co/qiyuw/WSPAlign-ft-enfr)| Finetuned with English-French dataset|
[qiyuw/WSPAlign-ft-roen](https://huggingface.co/qiyuw/WSPAlign-ft-roen)| Finetuned with Romanian-English dataset|

Use our model checkpoints with [huggingface](https://huggingface.co/)

## Data preparation
| Dataset list| Description|
|-------|-------|
|[qiyuw/qiyuw/wspalign_acl2023_eval](https://huggingface.co/datasets/qiyuw/wspalign_acl2023_eval)|Evaluation data used in the paper|

Construction of `Evaluation` dataset can be found at [word_align](https://github.com/nttcslab-nlp/word_align).

Run `download_dataset.sh` to download all the above datasets.

## Evaluation for WSPAlign Model.
The project also provides the evaluation script for pretrained and finetuned WSPAlign models, for details of the pre-training and fine-tuning of WSPAlign, please refer to [WSPAlign project](https://github.com/qiyuw/WSPAlign).

After running `finetune.sh`, `fewshot.sh` or `zeroshot.sh`, you will get the predicted alignment stored in `[YOUR OUTPUT DIR]/nbest_predictions_.json`. (e.g., `/data/local/qiyuw/WSPAlign/experiments-zeroshot-2023-08-03/zeroshot/deen/nbest_predictions_.json`)

Then run `cd evaluate` and `bash postprocess.sh [YOUR OUTPUT DIR]/nbest_predictions_.json [LANG] [TOKENIZER]`. The script will take care of the alignment transformation and evaluation. `[LANG]` can be chosen from `[deen, kftt, roen, enfr]`, and `[TOKENIZER]` can be chosen from `[BERT, ROBERTA]`.

See [evaluate/postprocess.sh](evaluate/postprocess.sh) for details.

## License

This software is released under the NTT License, see [LICENSE.txt](LICENSE.txt).