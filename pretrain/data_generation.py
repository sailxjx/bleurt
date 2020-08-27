import json
import pandas as pd
import re
import random
import math
import tensorflow as tf
import logging
from os import path
from transformers import AutoTokenizer, TFAutoModelWithLMHead
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import traceback, sys

import configparser
from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.acs_exception.exceptions import ClientException
from aliyunsdkcore.acs_exception.exceptions import ServerException
from aliyunsdkalimt.request.v20181012.TranslateGeneralRequest import TranslateGeneralRequest


logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

logger.info("Loading model ...")
tokenizer = AutoTokenizer.from_pretrained(
    "/home/admin/workspace/model/transformers/bert-base-multilingual-cased")
model = TFAutoModelWithLMHead.from_pretrained(
    "/home/admin/workspace/model/transformers/bert-base-multilingual-cased")

config = configparser.ConfigParser()
config.read("/home/admin/workspace/.secret")

client = AcsClient(config["account xjx"]["access_key"],
                   config["account xjx"]["access_secret"],
                   'cn-hangzhou')


def cut_sentences(text, min_len=3):
    """
    Cut sentences by their length and punctuation, remove all spaces.
    """
    text = text.replace(" ", "")
    corpus = re.split(r"[\,\.\?，。？\n]", text)
    corpus = list(filter(lambda x: len(x) >= min_len, corpus))
    return corpus


def mask_replacing(s):
    """
    The first strategy samples random words in the sentence and it replaces them with masks(one for each token).
    """
    seq = list(s)
    seq_len = len(s)
    # Sample from 1 to 90% chars of the sequence
    k = random.randint(1, math.floor(seq_len * 0.9))
    token_idx = random.choices(range(seq_len), k=k)
    for i in token_idx:
        seq[i] = "[MASK]"
    masked_rate = len(token_idx) / seq_len
    masked = "".join(seq)
    return pd.Series([masked, masked_rate], index=["masked", "masked_rate"])


def mask_replacing2(s):
    """
    The second strategy cre-ates contiguous sequences: 
    it samples a start po-sition s, a length l (uniformly distributed), 
    and it masks all the tokens spanned by words betweenpositions s and s + l.
    """
    seq_len = len(s)
    start = random.randint(1, seq_len-1)
    # At least 10% of words
    min_length = min(math.floor(seq_len * 0.1), seq_len - start)
    min_length = max(min_length, 1)
    # At most 90% of words
    max_length = min(math.floor(seq_len * 0.9), seq_len - start)
    max_length = max(min_length, max_length)
    length = random.choice(range(min_length, max_length+1))

    s = s[:start] + "[MASK]" * length + s[(start+length):]
    return pd.Series([s, length / seq_len], index=["masked", "masked_rate"])


def mask_filling(input_texts):
    encoded_input = tokenizer(input_texts, padding=True, return_tensors='tf')
    [predictions] = model(encoded_input)
    predicted_index = tf.argmax(predictions, axis=2)
    predicted_tokens = [tokenizer.convert_ids_to_tokens(index) for index in predicted_index]
    filled_seqs = ["".join(predict_token[1:np.sum(mask)-1]) \
        for [predict_token, mask] in zip(predicted_tokens, encoded_input["attention_mask"])]
    return filled_seqs

class BackTranslation:
    def __init__(self):
        self.bulk_size = 4800

    def back_translation(self, corpus):
        translated = self._bulk_translate(corpus, from_lang="zh", to_lang="en")
        back_translated = self._bulk_translate(
            translated, from_lang="en", to_lang="zh")
        return back_translated

    def _bulk_translate(self, corpus, from_lang="zh", to_lang="en"):
        translated = []
        text = ""

        def _do_translate(text, translated):
            translated_text = self._translate(
                text.strip(), from_lang=from_lang, to_lang=to_lang)
            translated += translated_text.split("\n")

        for seq in corpus:
            if len(text + seq) >= self.bulk_size:
                _do_translate(text, translated)
                text = seq + "\n"
            else:
                text += seq + "\n"

        _do_translate(text, translated)

        return translated

    def _translate(self, text, from_lang="zh", to_lang="en"):
        """
        The api of alimt has limit the maximum length of text to 5000 characters, maximum QPS to 50,
        so we should send the request in several bulks, with less than 250000 characters in each bulk.
        """
        request = TranslateGeneralRequest()
        request.set_accept_format('json')

        request.set_FormatType("text")
        request.set_SourceLanguage(from_lang)
        request.set_TargetLanguage(to_lang)

        request.set_SourceText(text)

        response = client.do_action_with_exception(request)
        response_json = json.loads(response)

        try:
            translated = response_json["Data"]["Translated"]
            return translated
        except:
            logger.error("Response error %s", response)
            raise Exception("Response error")


def parallelize(df, func):
    partitions = multiprocessing.cpu_count()
    df_splited = np.array_split(df, partitions)
    df_splited = Parallel(
        n_jobs=partitions
    )(delayed(func)(df) for df in df_splited)
    return np.concatenate(df_splited)


def word_dropping(text):
    """
    Randomly drop some words in the sequence
    """
    seq = list(text)
    text_len = len(text)
    k = random.choice([1] + list(range(1, int(text_len/3))))
    for i in random.choices(range(text_len), k=k):
        seq[i] = ""
    dropped_rate = k/text_len
    dropped = "".join(seq)
    return pd.Series([dropped, dropped_rate], index=["dropped", "dropped_rate"])


def make_candidates(references):
    """
    30% with mask filling rule1: scored by masked_rate
    30% with mask filling rule2: scored by masked_rate
    30% with back translation and word dropping: scored by dropped rate
    10% with back translation: score 0.98

    Returns
    -------
    candidates: Generated candidates with the same length of references
    scores: Arbitrary scores
    """
    # Do not modify input params
    references = references.copy()
    random.shuffle(references)
    refs = []

    ref_len = len(references)

    # Apply mask filling
    logger.info("Apply mask filling ...")
    mf1_len = mf2_len = int(ref_len*0.3)
    candidates = []
    scores = []

    # Mask filling
    mf1 = map(mask_replacing, references[:mf1_len])
    refs += references[:mf1_len]
    del references[:mf1_len]
    mf2 = map(mask_replacing2, references[:mf2_len])
    refs += references[:mf2_len]
    del references[:mf2_len]
    mf = pd.DataFrame(list(mf1) + list(mf2))
    candidates += mask_filling(mf.masked.tolist())
    scores += (1 - mf.masked_rate).values.tolist()

    # Back translation (bt)
    logger.info("Apply back translation ...")
    try:
        bt = parallelize(
            references, lambda refs: BackTranslation().back_translation(refs))
    except:
        # Error in back translation
        traceback.print_exc(file=sys.stdout)
        return refs, candidates, scores
    
    # Drop samples where refs and back translationed excactly same
    df_bt = pd.DataFrame({
        "refs": references,
        "bt": bt
    })
    df_bt = df_bt[df_bt.refs != df_bt.bt]

    # Replace references and bt for later use
    logger.info("Dropped {} samples".format(len(bt) - df_bt.shape[0]))
    refs += df_bt.refs.tolist()
    references = df_bt.refs.tolist()
    bt = df_bt.bt.tolist()

    # Apply 30% with word dropping
    wd_len = int(df_bt.shape[0] * 0.75)
    bt_dropped = map(word_dropping, bt[:wd_len])
    bt_dropped = pd.DataFrame(bt_dropped)
    candidates += bt_dropped.dropped.tolist()
    scores += (1 - bt_dropped.dropped_rate).tolist()

    del bt[:wd_len]
    candidates += bt
    scores += [0.98] * len(bt)

    return refs, candidates, scores


def save_data(dataset):
    """
    Save data to csv and jsonl
    jsonl example: {"candidate":"吴承恩是著名文学家","reference":"吴承恩是著名文学家","score":1}
    """
    csv_file = "./data_generationed/dataset.csv"
    jsonl_file = "./data_generationed/dataset.jsonl"

    mode = "w"
    if path.exists(csv_file):
        mode = "a"

    logger.info("Write to file {}".format(csv_file))
    dataset.to_csv(csv_file, index=None, header=None, mode=mode)

    def write_row(f, row):
        f.write(row.to_json(force_ascii=False) + "\n")

    logger.info("Write to file {}".format(jsonl_file))
    with open(jsonl_file, mode) as f:
        dataset.apply(lambda row: write_row(f, row), axis=1)


def readfile(filename, checkpoint = 0, batch_size = 300):
    content = []
    i = 0
    with open(filename, "r") as f:
        while True:
            i += 1
            if i > checkpoint and i <= (checkpoint + batch_size):
                line = f.readline()
                if line != "":
                    content.append(line)
                else:
                    checkpoint = None
                    break
            elif i <= checkpoint:
                next(f)
            else:
                checkpoint = i - 1
                break
    return content, checkpoint

def save_checkpoint(checkpoint):
    with open("./data_generationed/checkpoint", "w") as f:
        f.write("{}".format(checkpoint))

def run_epoch(filename, checkpoint = 0, batch_size = 1e4):
    logger.info("Loading dataset from checkpoint: {}".format(checkpoint))
    [content, checkpoint] = readfile(filename,
                                     checkpoint = checkpoint, 
                                     batch_size = batch_size)
    
    data = map(json.loads, content)
    data = pd.DataFrame(data)

    text = "\n".join(data.content.values)
    references = cut_sentences(text)
    logger.info("References count: {}".format(len(references)))

    [refs, candidates, scores] = make_candidates(references)

    dataset = pd.DataFrame({
        "reference": refs,
        "candidate": candidates,
        "score": scores
    })

    save_data(dataset)
    save_checkpoint(checkpoint)
    return checkpoint


def main():
    checkpoint = 0
    while True:
        checkpoint = run_epoch("./webtext2019zh/web_text_zh_train.json",
                               checkpoint = checkpoint,
                               batch_size = 100)
        if checkpoint is None:
            break
    logger.info("Finish")


if __name__ == "__main__":
    main()
