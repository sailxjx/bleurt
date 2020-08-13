#!/usr/bin/env python
#coding=utf-8
import configparser
from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.acs_exception.exceptions import ClientException
from aliyunsdkcore.acs_exception.exceptions import ServerException
from aliyunsdkalimt.request.v20181012.TranslateGeneralRequest import TranslateGeneralRequest
import numpy as np
import json
from joblib import Parallel, delayed
import multiprocessing

config = configparser.ConfigParser()
config.read("/home/admin/workspace/.secret")

client = AcsClient(config["account xjx"]["access_key"], 
                   config["account xjx"]["access_secret"], 
                   'cn-hangzhou')

class BackTranslation:
    def __init__(self):
        self.bulk_size = 4800
    
    def back_translation(self, corpus):
        translated = self._bulk_translate(corpus, from_lang = "zh", to_lang = "en")
        back_translated = self._bulk_translate(translated, from_lang = "en", to_lang = "zh")
        return back_translated
    
    def _bulk_translate(self, corpus, from_lang = "zh", to_lang = "en"):
        translated = []
        text = ""

        def _do_translate(text, translated):
            translated_text = self._translate(text.strip(), from_lang = from_lang, to_lang = to_lang)
            translated +=  translated_text.split("\n")
            
        for seq in corpus:
            if len(text + seq) >= self.bulk_size:
                _do_translate(text, translated)
                text = seq + "\n"
            else:
                text += seq + "\n"
                
        _do_translate(text, translated)
        
        return translated
    
    def _translate(self, text, from_lang = "zh", to_lang = "en"):
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

with open("./homebrewed/raw.txt") as f:
    corpus = f.readlines()
corpus = map(lambda x: x.strip(), corpus)
corpus = list(corpus)

back_translated = parallelize(corpus, lambda corpus: BackTranslation().back_translation(corpus))

with open("./homebrewed/back_translated.txt", "w") as f:
    for (s1, s2) in zip(corpus, back_translated):
        f.write("{}[SEP]{}\n".format(s1, s2))