from io import StringIO
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdevice import PDFDevice
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfpage import PDFTextExtractionNotAllowed
from pdfminer.pdfparser import PDFParser
from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from gensim import models
from logic.medidas import *
from tqdm import tqdm
import re
import chardet
import json
import numpy as np
import os
import pdfminer
import pickle
import time
from collections import defaultdict
from gensim import corpora
from gensim import similarities
# from gensim.test.utils import 
import spacy
from spacy.lang.es import Spanish , LOOKUP
from spacy.tokenizer import  Tokenizer
import nltk
from nltk.stem.snowball import SpanishStemmer


# from gensim.summarization import summarize



class RecuperationEngine():
    """Esta clase representa el core del motor de búsqueda 
    Basicamente crea el modelo vectorial o lo carga si ya lo construyo alguna vez
    y para una consulta se le pasa al metodo search_query este define si lo tiene indexado o no
    tranforma la consulta en un vector del spacio de la matrix de indices para luego poder compararlo con la la funcion de
    similaridad de coseno para poder ver que tanto se parece  a cada vector documento y luego generar un ranking y devolver los 1000
    más parecidos
    """
    def __init__(self):
        """ Aqui se instancia las estructuras que se basa el modelo 
            si es la primera ves se calcula el modela y la matrix de índeces
            sino se carga de la crapeta data
            Esta matiz se puede realizar gracias a los json guardaos producto del scrapeo. 

        """
        self.count= 0
        nlp = Spanish()
        self.tokenizer = Tokenizer(nlp.vocab)
        self.stemmer = SpanishStemmer()
        self.stopwordsfolder ='data/stopwords/'
        self.stopwords = []

        ##load stopwords
        for file in os.listdir(self.stopwordsfolder):
            with open(self.stopwordsfolder + file, encoding='utf-8') as fd:
                self.stopwords += fd.read().split()
        
        self.docs_prepoced =[]
        self.docs_preproces_gensim = []
        self.file_names = []
        self.datafolder = ''
        
        self.tf = TfidfVectorizer()
        
        val = self.load_retro_feed()
        if val == None:
            self.retro_feed_data = {}
        else:
            
            self.retro_feed_data= val
           

    def save_lsi_gsim(self):
        
        

        self.dictionary = corpora.Dictionary(self.docs_preproces_gensim)
        corpus = [self.dictionary.doc2bow(text) for text in self.docs_preproces_gensim]
        print('creating model..')
        self.lsi_model = models.LsiModel(corpus, id2word=self.dictionary, num_topics= 350)
        
        with open("data/lsi_model_gesim.pk",'wb') as pickle_file:
            pickle.dump(self.lsi_model, pickle_file)

        print('creating matrix ..')
        self.index = similarities.MatrixSimilarity(self.lsi_model[corpus])

        
        print('saving...matrix')
        self.index.save('data/index_lsi')


        

    def save_tfidf_matrix(self,):
        """En este metodo se llama si no se han construido nunca los indices 
        Siguiendo la idea del modelo vectorial se construye una matrix termino-documento
        donde los documentos son el campo page content del json que se genero cuando se escrapeo.

        A pratir de aqui se  apoya en sklearn que genera dicha matrix cuyos pesos son el tf-idf normalizado.
        Como podemos ver los documentos estan representados como vectores de pesos tfidf como plantea el modelo clasico vectorial.

        En este caso la matrix llega a tener dimensiones de  10000* 450000 y es una matri muy esparcida.
        """

        print('Computing Tf-Idf Matrix...')
        self.tfidfmatrix = self.tf.fit_transform( self.docs_prepoced)
        
        print('Saving Matrix...')
        np.save('data/Tf-Idf-matrix',self.tfidfmatrix) 

        print('Saving Model...')
        with open('data/TfIdfVectorizer.pk', 'wb') as f:
            pickle.dump(self.tf,f)

    def load_tfidf_matrix(self):

        """Metodo para cargar la matrix de pesos del modelo ya calculada con anterioridad"""
        self.tfidfmatrix= np.load('data/Tf-Idf-matrix.npy').all()
    
    def load_tf_vectorizer(self):
        """Metodo para cargar el modelo """
        with open('data/TfIdfVectorizer.pk', 'rb') as f:
            self.tf = pickle.load(f)
    def transformQueryGensim(self,query):
        vec_bow = self.dictionary.doc2bow(query.lower().split())
        return self.lsi_model[vec_bow]  # convert the query to LSI space

    def transformQery(self,query):
        """Permite transformar la query en un vector del espacio de los documentos seguiendo la conversion tfi- df dado el modelo 
        cargado y la matrix de indices. Esto es necesario para poder calcular el coseno entre la consulta y cada documento y asi generar el 
        ranking  """
        return self.tf.transform([query])
    
    def cosinesimilarity(self,query_vector):
        """Calcula el vector de similardidad entre los documentos y la query. Tener en cuenta que para estos se usa la multiplicacion de 
        matrices para lograr optimizar este proceso ya que si revisamos la fórmlula con es más que la multiplicación normalizada  del vector query y la 
        matrix de índices """
        return cosine_similarity(query_vector,self.tfidfmatrix)



    def rank(self, query_vector,matrix,k=100):
        """ Genera el rankink de los documentos más parecidos a la query en este caso indicado por el parametro k por default los 1000
        documentos más parecidos. Se usa la similaridad de coseno ya que es la que mejor resultados ha alcansado para este modelo"""
        result = cosine_similarity(query_vector,matrix)
        
        index_sorted = np.argsort(result)
        return (result,index_sorted[0][self.count-k: ])

    def search_query(self, query,model= 'vec'):
        """Método principal del motor de búsqueda primero revisa si es un url o no. 
        Si lo es revisa si ya lo tiene indexado lo busca yllama a rank sino lo manda scrapear y con  todo el preprocesado que lleva
        y llama a rank. Esto siempre despues de haber tranforamdo el vector de consulta al espacio de los documentos indexados.


        
        """
        k = 100
        init = time.time()
        query = self.preprocces(query,query= True)

        if model == 'lsi':
            return self.search_query_LSA(query)
        if model == 'lsi-gensim':
            query_vector = self.transformQueryGensim(query)
            sims = self.index[query_vector]
            result = np.argsort(sims)[self.count-k:]

            pages = []
            results = []
            
            for i in result:
                pages.append((round(sims[i],4), self.file_names[i]))
                results.append(self.file_names[i])
            
            pages.reverse()
            results.reverse()

            rr,nr,ri,precc,recb,f_med,f1_med,r_prec = (0,0,0,0,0,0,0,0)

            if not self.retro_feed_data.get(query) == None:
                rr , nr , ri = self.calc_rr_nr_ri(query,results)
                precc = precision(rr,ri)
                recb  = recall(rr,nr)
                f_med = f_medida(recb,precc)
                f1_med = f1_medida(recb,precc)
                r_prec = r_precision(k,rr )




            time_took =round(time.time()-init,3)
            print('Your search took ' + str(round(time.time()-init,3))+ ' seconds')
            
            return pages,time_took, precc, recb , f_med , f1_med , r_prec
            


        

        print('Vectorizing Query')
        query_vector = self.transformQery(query)

        result, n_Mayores = self.rank(query_vector,self.tfidfmatrix)



        pages =[ ]
        results =[]

        for i in n_Mayores:
            pages.append((round(result[0][i],3), self.file_names[i] ))
            results.append(self.file_names[i])

        pages.reverse()
        results.reverse()

        rr,nr,ri,precc,recb,f_med,f1_med,r_prec = (0,0,0,0,0,0,0,0)

        if not self.retro_feed_data.get(query) == None:
            rr , nr , ri = self.calc_rr_nr_ri(query,results)
            precc = precision(rr,ri)
            recb  = recall(rr,nr)
            f_med = f_medida(recb,precc)
            f1_med = f1_medida(recb,precc)
            r_prec = r_precision(k,rr )




        time_took =round(time.time()-init,3)
        print('Your search took ' + str(round(time.time()-init,3))+ ' seconds')
        
        return pages,time_took, precc, recb , f_med , f1_med , r_prec

    def calc_rr_nr_ri(self,query,results):
        """devuelve una tripla con 
            rr: relevantes recuperados
            nr: relevantes no recuperados
            ri: irrelevantes recuperados
         """
        results = set(results)
        relevants = set( self.retro_feed_data[query])
        rr = relevants.intersection(results)
        ri = results.difference(relevants)
        nr =  relevants.difference(results)

        return (len(rr),len(nr),len(ri))

    def preprocces(self,text,query = False):
            
        doc = re.split(r'\W+', text.lower())

        result = " "
        for word in doc:
            # print(word)
            if word in self.stopwords:
                continue
            value =  LOOKUP.get(word)
            if value == None:
                result+= " " + word
                continue

            result += " " + value


        if not query:
            self.docs_prepoced.append(result)
            # self.docs_preproces_gensim.append(result1)
        else:
            return result

    def load_folder(self, folder="data/"):
        print("load_folder")
        self.datafolder = folder
        self.file_names = []
        self.count = 0
        for file in os.listdir(self.datafolder):
            if file.endswith(".pdf") or file.endswith(".txt"):
                self.file_names.append(file)
                new_file = os.path.join(self.datafolder,file)
                self.count+=1
                if file.endswith(".txt"):
                    with open(new_file, encoding='utf-8') as fd:
                        # self.preprocces(fd.read())
                        text = fd.read()
                        self.docs_prepoced.append(text)
                        self.docs_preproces_gensim.append(text.split())

                        # if self.count %100 ==0:
                        #     # print(self.count)
                else:
                    self.preprocces(self.read_pdf(new_file))
                
    
    def read_pdf(self, string):
        rsrcmgr = PDFResourceManager()
        retstr = StringIO()
        codec ='utf-8'
        laparams = LAParams()
        device =  TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
        fp = open(string, 'rb')
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        password=''
        maxpages = 0
        caching=True
        pagenos = set()

        for page in PDFPage.get_pages(fp, pagenos, 
                                        maxpages=maxpages, 
                                        password=password, 
                                        caching=caching, 
                                        check_extractable=True):
            interpreter.process_page(page)

        text = retstr.getvalue()

        fp.close()
        device.close()
        retstr.close()
        # print(text)

        def fix_accents(string):
            string = string.replace("´ı", "í")
            string = string.replace("˜n", "ñ")
            string = string.replace("´a", "á")
            string = string.replace("´e", "é")
            string = string.replace("´o", "ó")
            string = string.replace("´u", "ú")
            # print(string)
            return string
        return fix_accents(text)

    def add_retro_feed(self,key,items, results):
        """añade a el diccionario el conjunto de documentos relevantes que marco el usuario 
        para el query key"""
        

        """ nota : revisar si lo dejamos como set o no , ya que el set no es json 
        serializable lo que hace que no podamos guardar este fees back en disco por lo menos no sin hacerle nada """
        # print(self.retro_feed_data)

        aux = []

        for i ,j in results:
            aux.append(j)


        if not self.retro_feed_data.get(key) == None:

            self.retro_feed_data[key].pop()
            for i in items:
                if not i in self.retro_feed_data[key]:
                    self.retro_feed_data[key].append(i)
            

            rr , nr , ri = self.calc_rr_nr_ri(key,aux)
            precc = precision(rr,ri)
            recb  = recall(rr,nr)
            f_med = f_medida(recb,precc)
            f1_med = f1_medida(recb,precc)
            r_prec = r_precision(10,rr )

            self.retro_feed_data[key].append([precc,recb,f1_med,f1_med,r_prec])
        else:
            # self.retro_feed_data[key] = set()
            self.retro_feed_data[key] = items

            rr , nr , ri = self.calc_rr_nr_ri(key,aux)
            precc = precision(rr,ri)
            recb  = recall(rr,nr)
            f_med = f_medida(recb,precc)
            f1_med = f1_medida(recb,precc)
            r_prec = r_precision(10,rr )

            self.retro_feed_data[key].append([precc,recb,f1_med,f1_med,r_prec])


    def load_retro_feed(self,path = 'data/retro_feed_data.json'):
        try:
            with open('data/retro_feed.json','r',encoding='utf-8')as fd:
                return json.load(fd)
        except:
            return None

    def save_retro_feed(self):

        save_dic = self.load_retro_feed()
        if save_dic == None:
            with open('data/retro_feed.json','w', encoding='utf-8')as f:
                # print(self.retro_feed_data)
                json.dump(self.retro_feed_data,f)
        else:
            # save_dic.update(self.retro_feed_data)
            # self.retro_feed_data= save_dic
            # print("ya existe")
            # print(self.retro_feed_data)
            with open('data/retro_feed.json','w', encoding='utf-8')as fk:
                json.dump(self.retro_feed_data,fk)

    def LSA(self, folder="data/"):
        self.svd = TruncatedSVD(n_components = 300)
        self.svdMatrix = self.svd.fit_transform(self.tfidfmatrix)
        # self.svdMatrix = Normalizer(copy=False).fit_transform(self.svdMatrix)

    def search_query_LSA(self, query):
        init = time.time()
        k = 100
        query_vector = self.transformQery(query)
        query_vector = self.svd.transform(query_vector)


        result, n_Mayores = self.rank(query_vector,self.svdMatrix)

        

        pages =[ ]
        results = []

        for i in n_Mayores:
            pages.append((round(result[0][i],3),self.file_names[i]))
            results.append(self.file_names[i])

        pages.reverse()
        results.reverse()

        rr,nr,ri,precc,recb,f_med,f1_med,r_prec = (0,0,0,0,0,0,0,0)

        if not self.retro_feed_data.get(query) == None:
            rr , nr , ri = self.calc_rr_nr_ri(query,results)
            precc = logic.medidas.precision(rr,ri)
            recb  = logic.medidas.recall(rr,nr)
            f_med = logic.medidas.f_medida(recb,precc)
            f1_med = logic.medidas.f1_medida(recb,precc)
            r_prec = logic.medidas.r_precision(k,rr )




        time_took =round(time.time()-init,3)
        print('Your search took ' + str(round(time.time()-init,3))+ ' seconds')
        
        return pages,time_took, precc, recb , f_med , f1_med , r_prec

    def save_LSA(self):
        np.save("svdMatrix", self.svdMatrix)
        with open("svdModel.pk",'wb') as pickle_file:
            pickle.dump(self.svd, pickle_file)
    def load_LSA(self):
        with open("svdModel.pk",'rb') as pickle_file:
            self.svd = pickle.load(pickle_file)
        self.svdMatrix = np.load("svdMatrix")
    
    def loop(self):
        while True:
            try:
                query = input("Enter query > ")
                out1 = self.search_query(query)
                out2 = self.search_query_LSA(query)

                pprint(out1)
                pprint(out2)
            except KeyboardInterrupt as e:
                break
                
            

# if __name__ == "__main__":

#     a = RecuperationEngine()

#     a.load_folder("testdata/")
#     print(a.docs_prepoced)
#     print(a.count)
#     a.save_tfidf_matrix()
    # a.LSA()
    # a.load_LSA()
    # a.save_LSA()