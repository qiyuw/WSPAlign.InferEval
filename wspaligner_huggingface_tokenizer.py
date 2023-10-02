#!/usr/bin/env python3

"""WSPAligner with huggingface's tokenizer.

The inference code for WSPAligner with huggingface's tokenizer.

Typical usage example:

    args = parse_arguments()
    word_aligner = WSPAligner(args)
    result = word_aligner.align_single_sentence_pair(args.src_text, args.tgt_text)
"""

import argparse
import collections
from typing import Dict, List, Tuple

import torch
from langdetect import detect
from tqdm.auto import tqdm
from transformers import AutoTokenizer, BatchEncoding, pipeline

from datasets import Dataset

AlignmentResult = Dict[str, List[float | int]]


class WSPAligner:
    """The WSPAligner inference class with huggingface tokenizers.

    Give two sentences or two lists of sentences, the WSPAligner will align the words or tokens (after tokenization) in the sentences.

    Attributes:
        args: The arguments for the WSPAligner.
        tokenizer: A tokenizer which we use.
        sep: A separator string which we use.
        pipe: A question answering pipeline instance used for word alignmnet.
    """

    def __init__(self, args: argparse.Namespace):
        """Initializes the instance based on tokenizer name and model name.

        Args:
            args: The arguments for the WSPAligner.
        """
        self.args = args

        # Load tokenizer
        if args.word_alignment_tokenizer_name is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                args.word_alignment_model_name_or_path
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                args.word_alignment_tokenizer_name
            )

        self.sep = " \u00b6 "  # Use ' ¶ ' (pilcrow sign) for context separator
        self.subword_prefix = (
            self._get_subword_prefix()
        )  # Get the subword prefix according to the tokenizer name

        # Use question-answering pipeline for prediction
        self.pipe = pipeline(
            "question-answering",
            model=args.word_alignment_model_name_or_path,
            device=args.word_aligner_device,
        )  # need to be optimized

    def align_sentence_pairs_from_dataset(
        self,
        dataset: Dataset,
        is_print_result: bool = False,
        is_allow_duplicated: bool = True,
        src_lang: str = None,
        tgt_lang: str = None,
        threshold: float = 0.9,
    ) -> List[AlignmentResult]:
        source_sentences = dataset[src_lang]
        target_sentences = dataset[tgt_lang]
        return self.align_sentence_pairs(
            source_sentences,
            target_sentences,
            is_print_result=is_print_result,
            is_allow_duplicated=is_allow_duplicated,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            threshold=threshold,
        )

    def align_sentence_pairs(
        self,
        source_sentences: List[str],
        target_sentences: List[str],
        use_dataset: bool = False,
        is_print_result: bool = False,
        is_allow_duplicated: bool = True,
        src_lang: str = None,
        tgt_lang: str = None,
        threshold: float = 0.9,
        batch_size: int = 1000,
    ) -> List[AlignmentResult]:
        """Batch align sentences.

        Args:
            source_sentences: A list of source sentences.
            target_sentences: A list of target sentences.
            is_print_result: Whether to print the result.

        Returns:
            A list of the result of word alignment.
        """

        assert len(source_sentences) == len(
            target_sentences
        ), "The number of source sentences and target sentences should be the same."

        len_sentences = len(source_sentences)
        questions = []
        contexts = []
        num_questions_contexts_per_instance = []
        src_tgt_tokens_per_instance = []
        for i in range(len_sentences):
            src_sentence = source_sentences[i]
            tgt_sentence = target_sentences[i]
            src_marked_sentences, src_tokens = self.mark_seperator_on_sentence(
                src_sentence, lang=src_lang
            )
            tgt_marked_sentences, tgt_tokens = self.mark_seperator_on_sentence(
                tgt_sentence, lang=tgt_lang
            )

            if self.args.use_token:
                src_sentence = "".join(src_tokens)
                tgt_sentence = "".join(tgt_tokens)

            len_src_marked_sentences = len(src_marked_sentences)
            len_tgt_marked_sentences = len(tgt_marked_sentences)

            for i in range(len_src_marked_sentences):
                questions.append(src_marked_sentences[i])
                contexts.append(tgt_sentence)
            for i in range(len_tgt_marked_sentences):
                questions.append(tgt_marked_sentences[i])
                contexts.append(src_sentence)

            num_questions_contexts_per_instance.append(
                [len_src_marked_sentences, len_tgt_marked_sentences]
            )
            src_tgt_tokens_per_instance.append([src_tokens, tgt_tokens])

        if use_dataset:
            dataset = Dataset.from_dict({"question": questions, "context": contexts})
            predictions = self.unidirectional_algin_from_dataset(
                dataset, batch_size=batch_size
            )
        else:
            predictions = self.unidirectional_align(questions, contexts)

        results = []
        current_sentences = 0
        for i, num_questions_contexts in enumerate(num_questions_contexts_per_instance):
            len_src_marked_sentences, len_tgt_marked_sentences = num_questions_contexts
            src_predictions = predictions[
                current_sentences : current_sentences + len_src_marked_sentences
            ]
            tgt_predictions = predictions[
                current_sentences
                + len_src_marked_sentences : current_sentences
                + len_src_marked_sentences
                + len_tgt_marked_sentences
            ]
            current_sentences += len_src_marked_sentences + len_tgt_marked_sentences

            src_tokens, tgt_tokens = src_tgt_tokens_per_instance[i]

            src_to_tgt = self.bidirectional_align(
                src_predictions,
                tgt_predictions,
                src_tokens,
                tgt_tokens,
                threshold=threshold,
            )

            if not is_allow_duplicated:
                src_to_tgt = self._filter_out_duplicated(src_to_tgt)

            if is_print_result:
                self.print_result(src_to_tgt, src_tokens, tgt_tokens)
            results.append(src_to_tgt)
        return results

    def align_single_sentence_pair(
        self,
        source_sentence: str,
        target_sentence: str,
        is_print_result: bool = True,
        is_allow_duplicated: bool = True,
        src_lang: str = None,
        tgt_lang: str = None,
        threshold: float = 0.9,
    ) -> AlignmentResult:
        """Align two sentences.

        Args:
            source_sentence: The source sentence.
            target_sentence: The target sentence.
            is_print_result: Whether to print the result.

        Returns:
            A dict of token index pair (source-target) as key, averaged score and frequency.
        """

        src_marked_sentences, src_tokens = self.mark_seperator_on_sentence(
            source_sentence, lang=src_lang
        )
        tgt_marked_sentences, tgt_tokens = self.mark_seperator_on_sentence(
            target_sentence, lang=tgt_lang
        )

        if self.args.use_token:
            target_sentence = "".join(tgt_tokens)
            source_sentence = "".join(src_tokens)

        len_src_marked_sentences = len(src_marked_sentences)
        len_tgt_marked_sentences = len(tgt_marked_sentences)
        questions = []
        contexts = []
        for i in range(len_src_marked_sentences):
            questions.append(src_marked_sentences[i])
            contexts.append(target_sentence)
        for i in range(len_tgt_marked_sentences):
            questions.append(tgt_marked_sentences[i])
            contexts.append(source_sentence)

        predictions = self.unidirectional_align(questions, contexts)
        src_predictions = predictions[:len_src_marked_sentences]
        tgt_predictions = predictions[len_src_marked_sentences:]

        src_to_tgt = self.bidirectional_align(
            src_predictions, tgt_predictions, src_tokens, tgt_tokens, threshold=0.9
        )

        if not is_allow_duplicated:
            src_to_tgt = self._filter_out_duplicated(src_to_tgt)

        if is_print_result:
            self.print_result(src_to_tgt, src_tokens, tgt_tokens)
        return src_to_tgt

    def mark_seperator_on_sentence(
        self, sentence: str, lang: str = None
    ) -> Tuple[List[str], List[str]]:
        """Prepare word alignment data. (insert seperator into the different positions of the sentence. / mark the word we want to align.)

        Args:
            sentence: The sentence whose words or tokens are what we want to align.

        Returns:
            A tuple of the marked sentences and tokens.
        """

        if self.args.use_token:
            be_no_special_tokens = self.tokenizer(
                sentence, add_special_tokens=False, return_special_tokens_mask=False
            )
            tokens = be_no_special_tokens.tokens()
        else:
            be = self.tokenizer(
                sentence, add_special_tokens=True, return_special_tokens_mask=True
            )
            word_token_table = self.get_xlm_word_token_table(be, lang=lang)
            tokens = self.assemble_words(word_token_table)

        # Detect the language of the sentence
        if lang is None:
            lang = detect(sentence)

        # Start marking the sentence.
        examples: List[str] = []
        joiner = "" if lang in ["ja", "zh-cn", "zh-tw"] else " "
        for i, token in enumerate(tokens):
            marked_token = f"{self.sep}{token}{self.sep}"
            example = joiner.join(tokens[:i] + [marked_token] + tokens[i + 1 :])
            examples.append(example)

        return examples, tokens

    def unidirectional_align(
        self, questions: List[str], contexts: List[str]
    ) -> List[Dict]:
        """Single-direction word alignment.

        Args:
            questions: A list of questions of the pipeline.
            context: A list of contexts of the pipeline. String type.

        Returns:
            A list of the preditions of the pipeline. each prediction is a dict which contains 'score', 'start', 'end' and 'answer' keys.

        Example:
            context: 'I like cats.'
            question: ' ¶ 私 ¶ は猫が好きです。'
        """
        return self.pipe(question=questions, context=contexts)

    def unidirectional_algin_from_dataset(
        self,
        dataset: Dataset,
        batch_size: int = 1000,
    ) -> List[Dict]:
        results = []
        for out in tqdm(self.pipe(dataset, batch_size=batch_size), total=len(dataset)):
            results.append(out)
        return results

    def bidirectional_align(
        self,
        src_predictions: List[Dict],
        tgt_predictions: List[Dict],
        src_tokens: List[str],
        tgt_tokens: List[str],
        threshold: float = 0.9,
        force_bidirectional: bool = False,
    ) -> AlignmentResult:
        """Bidirectional word alignment.

        Args:
            src_predictions: A list of the predictions from the source sentence.
            tgt_predictions: A list of the predictions from the target sentence.
            src_tokens: A list of tokens of the source sentence.
            tgt_tokens: A list of tokens of the target sentence.
            threshold: The threshold for filtering out low score word / token pairs.
            force_bidirectional: Whether filter out the word/token pairs that dont exist in two directions.

        Returns:
            A dict containing token index pair (source-target) as key, averaged score and frequency as values.
        """

        # source to target
        src_to_tgt: Dict[str, List[float | int]] = {}
        for i, src_pred in enumerate(src_predictions):
            # find the matching tokens from the target sentence tokens according to the prediction
            predicted_word_indices = self._find_matching_token(src_pred, tgt_tokens)
            for ith_tgt_word_idx in predicted_word_indices:
                word_pair = "{}-{}".format(i, ith_tgt_word_idx)
                src_to_tgt[word_pair] = [
                    src_pred["score"] / len(predicted_word_indices),
                    1,
                ]
                # divide by the number of predicted word idx to prevent too long spans from dominating,
                # one for counting if we have two predictions for the same word-pair.

        for i, tgt_pred in enumerate(tgt_predictions):
            predicted_word_indices = self._find_matching_token(tgt_pred, src_tokens)
            for ith_src_word_idx in predicted_word_indices:
                word_pair = "{}-{}".format(ith_src_word_idx, i)
                if word_pair in src_to_tgt:
                    src_to_tgt[word_pair][0] += tgt_pred["score"] / len(
                        predicted_word_indices
                    )
                    src_to_tgt[word_pair][1] += 1
                else:
                    continue

        # filter out word-pairs with low scores and unidirectional predictions
        src_to_tgt = {k: v for k, v in src_to_tgt.items() if v[0] > threshold}
        if force_bidirectional:
            src_to_tgt = {k: v for k, v in src_to_tgt.items() if v[1] > 1}
        return src_to_tgt

    def parse_alignment_results(
        self, results: List[AlignmentResult]
    ) -> List[Dict[int, int]]:
        """Parse the results of word alignment.

        Args:
            results: A list of the results of word alignment. Each result is a dict containing token index pair (source-target) as key, averaged score and frequency as values.

        Returns:
            A list of dicts. Each dict contains the source word index as key, and the target word index as value.
        """

        word_index_pairs = []
        for result in results:
            word_index_pairs.append(self.parse_alignment_result(result))
        return word_index_pairs

    def parse_alignment_result(self, result: AlignmentResult) -> Dict[int, int]:
        """Parse the result of word alignment.

        Args:
            result: The result of word alignment. AlignmentResult instance.

        Returns:
            A dict containing the source word index as key, and the target word index as value.
        """

        word_index_pair = {}
        for k, v in result.items():
            src_idx_str, tgt_idx_str = k.split("-")
            src_idx = int(src_idx_str)
            tgt_idx = int(tgt_idx_str)
            word_index_pair[src_idx] = tgt_idx
        return word_index_pair

    def print_result(
        self, result: AlignmentResult, src_tokens: List[str], tgt_tokens: List[str]
    ):
        """Print the result of word alignment.

        Args:
            result: The result of word alignment. AlignmentResult instance.
            src_tokens: a list of tokens of the source sentence.
            tgt_tokens: a list of tokens of the target sentence.
        """

        print("source2target:")
        print(f"{' '.join(src_tokens)} -> {' '.join(tgt_tokens)}")
        print()

        for k, v in result.items():
            src_idx_str, tgt_idx_str = k.split("-")
            src_idx = int(src_idx_str)
            tgt_idx = int(tgt_idx_str)
            src_token = src_tokens[src_idx]
            tgt_token = tgt_tokens[tgt_idx]
            print(f"{src_token}({src_idx}) - {tgt_token}({tgt_idx}) : {v}")

    def _filter_out_duplicated(self, result: AlignmentResult) -> AlignmentResult:
        """Filter out the duplicated word-pairs.

        If the source token idx has duplicated target token idx, we will only keep the one with the highest score.

        Args:
            result: The result of word alignment. AlignmentResult instance.

        Returns:
            The filtered result of word alignment.
        """

        source_token_indices = set()

        # Get unique source token indices
        for k, v in result.items():
            src_idx_str, tgt_idx_str = k.split("-")
            src_idx = int(src_idx_str)
            source_token_indices.add(src_idx)

        # Sort the source token indices
        source_token_indices = sorted(source_token_indices)

        new_result = {}
        # Get the highest score for each source token index
        for src_idx in source_token_indices:
            highest_score = 0
            highest_tgt_idx = -1
            frequency = 0
            for k, v in result.items():
                current_src_idx_str, current_tgt_idx_str = k.split("-")
                current_src_index = int(current_src_idx_str)
                current_tgt_idx = int(current_tgt_idx_str)
                if src_idx == current_src_index:
                    if v[0] > highest_score:
                        highest_score = v[0]
                        highest_tgt_idx, frequency = current_tgt_idx, v[1]
            new_result[f"{src_idx}-{highest_tgt_idx}"] = [highest_score, frequency]

        return new_result

    def _find_matching_token(self, src_pred: Dict, tgt_tokens: List[str]) -> List[int]:
        """Find the matching token from the target sentence tokens according to the prediction result.

        Args:
            src_pred: The prediction result from the source sentence.
            tgt_tokens: A list of tokens of the target sentence.

        Returns:
            A list of the matched token indices.
        """

        predicted_token = src_pred["answer"]
        marched_token_indices = []

        for i, tgt_token in enumerate(tgt_tokens):
            if tgt_token in predicted_token:
                marched_token_indices.append(i)

        return marched_token_indices

    def create_dataset_from_sentence_pairs(
        self,
        src_sentences: List[str],
        tgt_sentences: List[str],
        src_lang: str = None,
        tgt_lang: str = None,
    ) -> Dataset:
        """Create a dataset from sentence pairs."""

        return Dataset.from_dict({src_lang: src_sentences, tgt_lang: tgt_sentences})

    # def _tokens2words(self, tokens: List[str]) -> List[str]:
    #     """Convert tokens to words.

    #     Args:
    #         tokens: A list of the tokens of the sentence.

    #     Returns:
    #         A list of words.
    #     """

    #     if self.args.word_alignment_tokenizer_name is None:
    #         name = self.args.word_alignment_model_name_or_path
    #     else:
    #         name = self.args.word_alignment_tokenizer_name

    #     if "xlm" in name:
    #         subword_prefix = "▁"  # Notice: this prefix is different from the regular underline symbol "_".
    #     elif "mbert" in name:
    #         subword_prefix = "##"

    #     words = []
    #     for i, token in enumerate(tokens):
    #         if i == 0:
    #             if "xlm" in name:
    #                 words.append(token[len(subword_prefix) :])
    #             elif "mbert" in name:
    #                 words.append(token)
    #             continue
    #         elif token.startswith(subword_prefix):
    #             words.append(token[len(subword_prefix) :])
    #         else:
    #             lang = detect(self.tokenizer.convert_tokens_to_string(tokens))
    #             if lang in ["ja", "zh-cn", "zh-tw"]:
    #                 words.append(token)
    #             else:
    #                 words[-1] += token[:]
    #     return words

    def get_xlm_word_token_table(
        self, be: BatchEncoding, lang: str = None
    ) -> Dict[int, List[Dict[str, List[int]]]]:
        """Convert tokens to words.

        Convert tokens to words and get the mapping from word index to token indices.

        Args:
            be: Tokenizer result. BatchEncoding instance.

        Returns:
            A dict of the mapping from word index to token indices.
            Keys are the word indices.
            Values are dicts whose keys are the token strings and values are lists which contains two values, and the first value is the token index in the token list, and the second value is the id of the token.

        Example:
            tokens: ['<s>', '▁I', '▁like', '▁cat', 's', '.', '</s>']
            result: {0: [{'I': [1, 87]}], 1: [{'like': [2, 1884]}], 2: [{'cat': [3, 7515]}, {'s': [4, 7]}, {'.': [5, 5]}]}

            tokens: ['<s>', '_', '撒', '哈', '拉', '沙漠', '是', '非洲', '大陆', '最大的', '荒', '漠', '。', '</s>']
            result: {0: [{'撒': [2, 79007]}], 1: [{'哈': [3, 16659]}], 2: [{'拉': [4, 3300]}], 3: [{'沙漠': [5, 196064]}], 4: [{'是': [6, 354]}], 5: [{'非洲': [7, 76938]}], 6: [{'大陆': [8, 64940]}], 7: [{'最大的': [9, 19049]}], 8: [{'荒': [10, 44147]}], 9: [{'漠': [11, 155149]}], 10: [{'。': [12, 30]}]}
        """

        words = collections.defaultdict(list)
        word_index = 0
        tokens = be.tokens()
        input_ids = be.input_ids
        for token_idx, token in enumerate(tokens):
            if token_idx == 0:
                # First token
                if be.special_tokens_mask[token_idx] == 0:
                    if token.startswith(self.subword_prefix):
                        words[word_index] = []
                        words[word_index].append(
                            {
                                token[len(self.subword_prefix) :]: [
                                    token_idx,
                                    input_ids[token_idx],
                                ]
                            }
                        )
                    else:
                        words[word_index] = []
                        words[word_index].append(
                            {token: [token_idx, input_ids[token_idx]]}
                        )
                    word_index += 1
                continue
            elif token.startswith(self.subword_prefix):
                if token == self.subword_prefix:
                    continue
                # Encounter new word
                if be.special_tokens_mask[token_idx] == 0:
                    words[word_index] = []
                    words[word_index].append(
                        {
                            token[len(self.subword_prefix) :]: [
                                token_idx,
                                input_ids[token_idx],
                            ]
                        }
                    )
                    word_index += 1
            else:
                if lang is None:
                    lang = detect(self.tokenizer.convert_tokens_to_string(tokens))

                # Special case for Japanese and Chinese
                if lang in ["ja", "zh-cn", "zh-tw", 'zh']:
                    if be.special_tokens_mask[token_idx] == 0:
                        words[word_index] = []
                        words[word_index].append(
                            {token: [token_idx, input_ids[token_idx]]}
                        )
                        word_index += 1
                else:
                    if be.special_tokens_mask[token_idx] == 0:
                        words[word_index - 1].append(
                            {token: [token_idx, input_ids[token_idx]]}
                        )
        return words

    def assemble_words(self, word_token_table: Dict):
        """Assemble words.

        Args:
            word_token_table: A dict of the mapping from word index to token indices.
            Keys are the word indices.
            Values are dicts whose keys are the token strings and values are the corresponding token indices.

        Returns:
            A list of words.
        """

        words = []
        for word_idx, token_list in word_token_table.items():
            word = ""
            for token in token_list:
                word += list(token.keys())[0]
            words.append(word)
        return words

    # def get_words_and_word_indices_to_token_indices(
    #     self, tokens: List[str], special_tokens_mask: List[int]
    # ) -> Tuple[Dict[int, str], DefaultDict[int, List[int]]]:
    #     """Convert tokens to words.

    #     Args:
    #         tokens: A list of the tokens of the sentence.

    #     Returns:
    #         A list of words.
    #     """

    #     if self.args.word_alignment_tokenizer_name is None:
    #         name = self.args.word_alignment_model_name_or_path
    #     else:
    #         name = self.args.word_alignment_tokenizer_name

    #     if "xlm" in name:
    #         subword_prefix = "▁"  # Notice: this prefix is different from the regular underline symbol "_".
    #     elif "mbert" in name:
    #         subword_prefix = "##"

    #     words = []
    #     word_index = 0
    #     word_indices_to_token_indices = collections.defaultdict(
    #         list
    #     )  # word idx -> list of token indices
    #     for token_idx, token in enumerate(tokens):
    #         if token_idx == 0:
    #             # First token
    #             if special_tokens_mask[token_idx] == 0:
    #                 if "xlm" in name:
    #                     words.append(token[len(subword_prefix) :])
    #                 elif "mbert" in name:
    #                     words.append(token)
    #                 word_indices_to_token_indices[word_index].append(token_idx)
    #                 word_index += 1
    #             continue
    #         elif token.startswith(subword_prefix):
    #             if special_tokens_mask[token_idx] == 0:
    #                 words.append(token[len(subword_prefix) :])
    #                 word_indices_to_token_indices[word_index].append(token_idx)
    #                 word_index += 1
    #         else:
    #             lang = detect(self.tokenizer.convert_tokens_to_string(tokens))
    #             if lang in ["ja", "zh-cn", "zh-tw"]:
    #                 if special_tokens_mask[token_idx] == 0:
    #                     words.append(token)
    #                     word_indices_to_token_indices[word_index].append(token_idx)
    #                     word_index += 1
    #             else:
    #                 if special_tokens_mask[token_idx] == 0:
    #                     words[-1] += token[:]
    #                     word_indices_to_token_indices[word_index - 1].append(token_idx)

    #     return words, word_indices_to_token_indices

    def _get_subword_prefix(self):
        if self.args.word_alignment_tokenizer_name is None:
            name = self.args.word_alignment_model_name_or_path
        else:
            name = self.args.word_alignment_tokenizer_name

        if "xlm" in name:
            subword_prefix = "\u2581"  # Notice: this prefix is different from the regular underline symbol "_".
        elif "mbert" in name:
            subword_prefix = "##"
        return subword_prefix


def parse_arguments() -> argparse.Namespace:
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--word_alignment_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--word_alignment_tokenizer_name",
        type=str,
        default=None,
        help="The tokenizer to use.",
    )
    parser.add_argument(
        "--src_text",
        type=str,
        default="足利義満（あしかがよしみつ）は室町幕府の第3代征夷大将軍（在位1368年-1394年）である。",
        required=True,
        help="Source text to align.",
    )
    parser.add_argument(
        "--tgt_text",
        type=str,
        default="yoshimitsu ashikaga was the 3rd seii taishogun of the muromachi shogunate and reigned from 1368 to 1394.",
        required=True,
        help="Target text to align.",
    )
    parser.add_argument("--src_lang", type=str, default=None, help="Source language.")
    parser.add_argument("--tgt_lang", type=str, default=None, help="Target language.")
    parser.add_argument(
        "--use_token",
        action="store_true",
        help="Whether to use token alignment or word alignment.",
    )
    parser.add_argument(
        "--allow_duplicated",
        action="store_true",
        help="Whether to allow duplicated word-pairs.",
    )
    parser.add_argument(
        "--word_aligner_device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()
    return args


def main():
    # Entry point
    args = parse_arguments()
    word_aligner = WSPAligner(args)
    text = '因此,我谨要求各位成员真诚努力，使其发言尽量简要 -- -- 无论如何不超过10分钟 -- -- 以考虑到其他国家的代表团，以便在这些全体会议范围内照顾所有发言者。'
    word_aligner.get_xlm_word_token_table(word_aligner.tokenizer(text, add_special_tokens=True, return_special_tokens_mask=True), lang='zh')
    word_aligner.align_sentence_pairs(
        source_sentences=["私は猫が好きです", "私は犬が好きです"],
        target_sentences=["I like cats", "I like dogs"],
        use_dataset=True,
        is_print_result=True,
        is_allow_duplicated=args.allow_duplicated,
        threshold=0.9,
        batch_size=1000,
    )


if __name__ == "__main__":
    main()
