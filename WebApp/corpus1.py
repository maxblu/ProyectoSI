from logic.medidas import *
from logic.mainClass import *
import json
from pprint import pprint
import random

corpus_doc_path = "data/medicina docs [BIG]/"
# corpus_query_path = "data/medicina query/query1.txt"
corpus_all_query_path = "data/medicina query/all.txt"
corpus1_stats = "data/corpus1_stats.json"

def text_corpus_1():
	r = RecuperationEngine(BASE_DIR =corpus_doc_path,numTopics=200,rank=10)
	# r.load_tf_vectorizer()
	r.load_lsi_model()

	test_cases = test_all_queries()

	out_dict = {}

	for q1 in test_cases:
		# print(q1)
		q = r.preprocces(q1, query=True)
		
		out_vec = [b for a, b in r.search_query(q, model= 'vec')[0]]
		out_lsi = [b for a, b in r.search_query(q, model= 'lsi-gensim')[0]]
		
		out_dict[q1] = {}
		out_dict[q1]["rel"] = test_cases[q1]["rel"]

		out_dict[q1]["rec-vec"] = out_vec
		out_dict[q1]["rec-lsi"] = out_lsi

		out_dict[q1]["rr-vec"], out_dict[q1]["nr-vec"], out_dict[q1]["ri-vec"] = RR_NR_RI(out_vec, test_cases[q1]["rel"])
		out_dict[q1]["rr-lsi"], out_dict[q1]["nr-lsi"], out_dict[q1]["ri-lsi"] = RR_NR_RI(out_lsi, test_cases[q1]["rel"])

		m1 = measures(RR_NR_RI(out_vec, test_cases[q1]["rel"]))
		m2 = measures(RR_NR_RI(out_lsi, test_cases[q1]["rel"]))
		
		if 0 in m1 + m2:
			# print("Got a zero", q1)
			del(out_dict[q1])
			# pprint(out_dict)
			# input()
			continue

		out_dict[q1]["measures-vec"] = m1
		out_dict[q1]["measures-lsi"] = m2

	return out_dict
	
def RR_NR_RI(recovered, relevants):
	recovered = set(recovered)
	relevants = set(relevants)
	rr = relevants.intersection(recovered)
	nr =  relevants.difference(recovered) 
	ri = recovered.difference(relevants)
	
	return list(rr), list(nr), list(ri)

def measures(t, r=20):
			(rr, nr, ri) = t
			rr = len(rr)
			nr = len(nr)
			ri = len(ri)
			precc = precision(rr, ri)
			recb  = recall(rr, nr)
			f_med = f_medida(recb, precc)
			f1_med = f1_medida(recb, precc)
			r_prec = r_precision(r, rr)
			return [precc,recb,f_med,f1_med,r_prec]

def load_query_dict():
	with open(corpus_query_path, encoding='utf-8') as fd:
		test_cases = json.load(fd)
	# pprint(test_cases)
	new_dict = {}
	for case in test_cases:
		query = case["Text"]
		rel = case["RelevantDocuments"]
		total = case["TotalDocuments"]

		new_dict[query] = rel + [eval()]

	with open("data/corpus1_stats.json",'w', encoding='utf-8') as fd:
		json.dump(new_dict, fd)

	new_dict = {}
	for case in test_cases:
		query = case["Text"]
		rel = case["RelevantDocuments"]
		total = case["TotalDocuments"]

		new_dict[query] = {}
		new_dict[query]["rel"] = rel
		new_dict[query]["measures"] = [eval() for x in range(5)]

	return new_dict

def test_all_queries():
	with open(corpus_all_query_path, encoding='utf-8') as fd:
		test_cases = json.load(fd)
	new_dict = {}
	for case in test_cases:
		query = case["Text"]
		rel = case["RelevantDocuments"]
		total = case["TotalDocuments"]

		new_dict[query] = {}
		new_dict[query]["rel"] = rel
		new_dict[query]["measures"] = [eval() for x in range(5)]


		# pprint((query,new_dict[query]))

	return new_dict

# DUMMY FUNCTIONS
def eval():
	return [random.random() for x in range(5)]
# class RecuperationEngine():
# 	"""docstring for DUMMY RecuperationEngine"""
# 	def __init__(self, BASE_DIR):
# 		self.BASE_DIR = BASE_DIR
# 		# print(self.BASE_DIR)

# 	def load_lsi_model(self):
# 		# print("LSI")
# 		pass
		
# 	def preprocces(self, text, query):
# 		# print(text)
# 		return text

# 	def search_query(self, query, model):
# 		# print(query, model)
# 		return []
		
		
		

# DUMMY FUNCTIONS
if __name__ == '__main__':
	d = text_corpus_1()

	# pprint(d)

	with open("data/corpus1_full_stats.json",'w', encoding='utf-8') as fd:
		json.dump(d, fd)
