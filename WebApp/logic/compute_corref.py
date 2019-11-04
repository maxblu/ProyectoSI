from gensim import corpora
from gensim import corpora
from  gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt

def compute_coh(dictionary, corpus,doc_clen,model):

    coherence_value = 0

    coherenceMOdel= CoherenceModel(model=model, texts=doc_clen, dictionary=dictionary,coherence ='c_v')
    return coherenceMOdel.get_coherence()