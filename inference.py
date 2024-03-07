import torch
import argparse
import transformers
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    pipeline
)
from spacy.lang.en import English
from spacy.lang.ja import Japanese
from spacy.lang.fr import French
from spacy.lang.de import German
from spacy.lang.zh import Chinese
from spacy.lang.ro import Romanian

context_sep =' \u00b6 ' # use ' ¶ ' (pilcrow sign) for context separator
language_tokenizer_map = {
        'en': English,
        'ja': Japanese,
        'fr': French,
        'de': German,
        'zh': Chinese,
        'ro': Romanian,
    } # map of language to tokenizer

def get_tokenizer(args):
    if args.src_lang and args.src_lang not in language_tokenizer_map:
        raise ValueError("Language {} not supported by SpaCy tokenizer.".format(args.src_lang))
    if args.tgt_lang and args.tgt_lang not in language_tokenizer_map:
        raise ValueError("Language {} not supported by SpaCy tokenizer.".format(args.tgt_lang))
    src_lang = args.src_lang
    tgt_lang = args.tgt_lang
    # special case for Chinese
    if src_lang in ['zh-cn', 'zh-tw']:
        src_lang = 'zh'
    if tgt_lang in ['zh-cn', 'zh-tw']:
        tgt_lang = 'zh'
    if src_lang not in language_tokenizer_map:
        src_lang = 'en'
    if tgt_lang not in language_tokenizer_map:
        tgt_lang = 'en'
    src_tokenizer_to_load = language_tokenizer_map[src_lang]()
    tgt_tokenizer_to_load = language_tokenizer_map[tgt_lang]()
    src_tokenizer = src_tokenizer_to_load.tokenizer
    tgt_tokenizer = tgt_tokenizer_to_load.tokenizer
    return src_tokenizer, tgt_tokenizer, src_lang, tgt_lang

def make_word_alignments_data(sent, tokenizer, context_sep=' \u00b6 '):
    words = tokenizer(sent)
    examples = []
    for word in words:
        example = sent[:word.idx] + context_sep + sent[word.idx:word.idx+len(word.text)] + context_sep + sent[word.idx+len(word.text):]
        examples.append(example)
    return examples, words

def align(pipe, src_examples, tgt_text, threshold=0.5):
    src_predictions = []
    for src_example in src_examples:
        pred = pipe(question=src_example, context=tgt_text)
        src_predictions.append(pred)
    return src_predictions

def find_matching_word(src_pred, tgt_words):
    start = src_pred['start']
    end = src_pred['end']
    # filter out tgt_words that are not in the predicted span
    predicted_word_idx = []
    for i, tgt_word in enumerate(tgt_words):
        if tgt_word.idx >= start and tgt_word.idx+len(tgt_word.text) <= end:
            predicted_word_idx.append(i)
    return predicted_word_idx

def bidirectional_align(src_predictions, tgt_predictions, src_words, tgt_words, threshold=0.1, force_bidirectional=False):
    src_to_tgt = {}
    for i, src_pred in enumerate(src_predictions):
        predicted_word_idx = find_matching_word(src_pred, tgt_words)
        for ith_tgt_word in predicted_word_idx:
            word_pair = "{}-{}".format(i, ith_tgt_word)
            src_to_tgt[word_pair] = [src_pred['score'] / len(predicted_word_idx), 1]
            # divide by the number of predicted word idx to prevent too long spans from dominating,
            # one for counting if we have two predictions for the same word-pair.
    for i, tgt_pred in enumerate(tgt_predictions):
        predicted_word_idx = find_matching_word(tgt_pred, src_words)
        for ith_src_word in predicted_word_idx:
            word_pair = "{}-{}".format(ith_src_word, i)
            if word_pair in src_to_tgt:
                src_to_tgt[word_pair][0] += tgt_pred['score'] / len(predicted_word_idx)
                src_to_tgt[word_pair][1] += 1
            else:
                continue
    # filter out word-pairs with low scores and unidirectional predictions
    src_to_tgt = {k: v for k, v in src_to_tgt.items() if v[0] > threshold}
    if force_bidirectional:
        src_to_tgt = {k: v for k, v in src_to_tgt.items() if v[1] > 1}
    return src_to_tgt

def print_alignments(src_text, tgt_text, src_words, tgt_words, src_to_tgt, src_lang, tgt_lang):
    print("Source ({}) text: {}".format(src_lang, src_text))
    print("Target ({}) text: {}".format(tgt_lang, tgt_text))
    for src_tgt in src_to_tgt:
        src_idx, tgt_idx = src_tgt.split('-')
        src_idx = int(src_idx)
        tgt_idx = int(tgt_idx)
        src_word = src_words[src_idx]
        tgt_word = tgt_words[tgt_idx]
        print("{} ---- {} | Score: {}".format(src_word, tgt_word, src_to_tgt[src_tgt][0]))

def align_one_example(pipe, src_text, tgt_text, src_tokenizer, tgt_tokenizer, print_results=True):
    src_examples, src_words = make_word_alignments_data(src_text, src_tokenizer)
    tgt_examples, tgt_words = make_word_alignments_data(tgt_text, tgt_tokenizer)

    src_predictions = align(pipe, src_examples, tgt_text)
    tgt_predictions = align(pipe, tgt_examples, src_text)

    alignments = bidirectional_align(src_predictions, tgt_predictions, src_words, tgt_words)

    if print_results:
        print_alignments(src_text, tgt_text, src_words, tgt_words, alignments, src_lang, tgt_lang)
    return alignments

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument("--src_lang", type=str, required=True, help="Language of the SpaCy tokenizer to use.")
    parser.add_argument("--tgt_lang", type=str, required=True, help="Language of the SpaCy tokenizer to use.")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument("--src_text", type=str, required=True, help="Source text to align.")
    parser.add_argument("--tgt_text", type=str, required=True, help="Target text to align.")
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    src_text = args.src_text
    tgt_text = args.tgt_text
    # examples
    # src_text = "私は猫が好きです。"
    # tgt_text = "I like cats."
    # src_text = "足利義満（あしかがよしみつ）は室町幕府の第3代征夷大将軍（在位1368年-1394年）である。"
    # tgt_text = "yoshimitsu ashikaga was the 3rd seii taishogun of the muromachi shogunate and reigned from 1368 to 1394."
    # src_text = "要拿到一个博士学位是需要付出很多努力的。"
    # tgt_text = "getting a phd requires a lot of hard work."

    # SpaCy tokenizer
    src_tokenizer, tgt_tokenizer, src_lang, tgt_lang = get_tokenizer(args)

    # use question-answering pipeline for prediction
    pipe = pipeline("question-answering", model=args.model_name_or_path)

    align_one_example(pipe, src_text, tgt_text, src_tokenizer, tgt_tokenizer)