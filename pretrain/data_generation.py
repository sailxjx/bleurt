import json
import pandas as pd
import re
import random
import math
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelWithLMHead


def cut_sentences(text, min_len=3):
    """
    Cut sentences by their length and punctuation, remove all spaces.
    """
    text = text.replace(" ", "")
    corpus = re.split("[\,\.\?，。？\n]", text)
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


tokenizer = AutoTokenizer.from_pretrained(
    "/home/admin/workspace/model/transformers/bert-base-multilingual-cased")
model = TFAutoModelWithLMHead.from_pretrained(
    "/home/admin/workspace/model/transformers/bert-base-multilingual-cased")


def mask_filling(text):
    encoded_input = tokenizer(text, return_tensors='tf')
    [predictions] = model(encoded_input)

    predicted_index = tf.argmax(predictions[0], axis=1)
    predicted_token = tokenizer.convert_ids_to_tokens(predicted_index)
    return "".join(predicted_token[1:-1])


import configparser
from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.acs_exception.exceptions import ClientException
from aliyunsdkcore.acs_exception.exceptions import ServerException
from aliyunsdkalimt.request.v20181012.TranslateGeneralRequest import TranslateGeneralRequest

config = configparser.ConfigParser()
config.read("/home/admin/workspace/.secret")

client = AcsClient(config["account xjx"]["access_key"],
                   config["account xjx"]["access_secret"],
                   'cn-hangzhou')

import numpy as np
import json
from joblib import Parallel, delayed
import multiprocessing


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
            print(response_json)
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


with open("./webtext2019zh/web_text_zh_train_sample.json", "r") as f:
    content = f.readlines()

data = map(json.loads, content)
data = pd.DataFrame(data)

text = "\n".join(data.content.values)
references = cut_sentences(text)

import logging
logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


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
    mf_filled = mf.masked.apply(mask_filling)

    candidates = mf_filled.tolist()
    scores += (1 - mf.masked_rate).values.tolist()
    
    # Back translation (bt)
    logger.info("Apply back translation ...")
    bt = parallelize(references, lambda refs: BackTranslation().back_translation(refs))
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


[refs, candidates, scores] = make_candidates(references)

dataset = pd.DataFrame({
    "reference": refs,
    "candidate": candidates,
    "score": scores
})

filename = "./data_generationed/dataset.csv"
logger.info("Write to file {}".format(filename))
dataset.to_csv(filename, index=None)
