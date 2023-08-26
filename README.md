# WSPAligner-inference
This project provides the inference libarary for [WSPAlign](https://github.com/qiyuw/WSPAlign).

## Requirements
Run `pip install -r requirements.txt` to install the required packages.

## Evaluation for WSPAlign Model.
The project also provides the evaluation script for pretrained and finetuned WSPAlign models, for details of the pre-training and fine-tuning of WSPAlign, please refer to [WSPAlign project](https://github.com/qiyuw/WSPAlign).

After running `finetune.sh`, `fewshot.sh` or `zeroshot.sh`, you will get the predicted alignment stored in `[YOUR OUTPUT DIR]/nbest_predictions_.json`. (e.g., `/data/local/qiyuw/WSPAlign/experiments-zeroshot-2023-08-03/zeroshot/deen/nbest_predictions_.json`)

Then run `cd evaluate` and `bash postprocess.sh [YOUR OUTPUT DIR]/nbest_predictions_.json [LANG] [TOKENIZER]`. The script will take care of the alignment transformation and evaluation. `[LANG]` can be chosen from `[deen, kftt, roen, enfr]`, and `[TOKENIZER]` can be chosen from `[BERT, ROBERTA]`.

See [evaluate/postprocess.sh](evaluate/postprocess.sh) for details.

## License

This software is released under the NTT License, see [LICENSE.txt](LICENSE.txt).