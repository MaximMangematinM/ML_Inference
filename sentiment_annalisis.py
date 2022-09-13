import torch
from transformers import pipeline
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from scipy.special import softmax




pipe = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

def eval_sentence_pipeline_roberta_twitter(sentence) -> str:
    result = {"LABEL_0" : "NEGATIVE", "LABEL_1" : "NEUTRAL", "LABEL_2" : "POSITIVE"}
    return result[pipe(sentence)[0]["label"]]



sentiment_analysis_roberta = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english")

def eval_sentence_roberta_large(sentence)-> str:
    return sentiment_analysis_roberta(sentence)[0]["label"]