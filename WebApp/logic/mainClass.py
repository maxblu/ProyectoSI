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
import gensim
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
from gensim.summarization import keywords
from gensim import corpora
from logic.compute_corref import *
from gensim import similarities
# from gensim.test.utils import 
import spacy
from spacy.lang.es import Spanish , LOOKUP
from spacy.lang.en import LOOKUP as enL
from spacy.lang.en import  English
from spacy.tokenizer import  Tokenizer
import nltk
from nltk.stem.snowball import SpanishStemmer


# from gensim.summarization import summarize

def chronodecorator(func):
    def wrapper(*arg, **kwargs):
    
                t = time.process_time()
                print("%27s Function: %10s" % (str(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())),func.__name__,))
                res = func(*arg, **kwargs)
                print("%27s Function: %10s Time: %4f Output: %20s"  % (str(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())),func.__name__,time.process_time()-t, str(res)[:min(len(str(res)),20)]))
                return res
	
    return wrapper




class RecuperationEngine():
    """Esta clase representa el core del motor de búsqueda 
    Basicamente crea el modelo vectorial o lo carga si ya lo construyo alguna vez
    y para una consulta se le pasa al metodo search_query este define si lo tiene indexado o no
    tranforma la consulta en un vector del spacio de la matrix de indices para luego poder compararlo con la la funcion de
    similaridad de coseno para poder ver que tanto se parece  a cada vector documento y luego generar un ranking y devolver los 1000
    más parecidos
    """
    def __init__(self,model='vec', BASE_DIR = 'data/',numTopics= 350,rank= 20):
        """ Aqui se instancia las estructuras que se basa el modelo 
            si es la primera ves se calcula el modela y la matrix de índeces
            sino se carga de la crapeta data
            Esta matiz se puede realizar gracias a los json guardaos producto del scrapeo. 

        """
        if  numTopics == None:
            self.numTopics = 350
        else:
            self.numTopics = numTopics

        self.numTopics= numTopics
        self.count= 0
        self.model= model
        nlp = Spanish()
        # nlpen = English()
        if rank == None:
            self.k = 20
        else:
            self.k = rank
        self.tokenizer = Tokenizer(nlp.vocab)
        # self.tokenizer_en = Tokenizer(nlpen)
        # self.stemmer = SpanishStemmer()
        self.stopwordsfolder ='data/stopwords/'
        self.stopwords = []
        self.doc_to_index={}
        self.query_opt = {}
        self.relevantes={}
        ##load stopwords
        for file in os.listdir(self.stopwordsfolder):
            with open(self.stopwordsfolder + file, encoding='utf-8') as fd:
                self.stopwords += fd.read().split()
        self.docs_prepoced =[]
        self.docs_preproces_gensim = []
        self.file_names = []
        self.datafolder = ''
        self.load_folder(BASE_DIR)
        # self.docs_index = {}
        self.load_relevant()

        if model == 'vec':
            self.load_tf_vectorizer()
        elif model == "lsi-gensim":
            self.load_lsi_model()
        else:
            self.load_lsi_model()
            self.load_tf_vectorizer()


        # print(self.index)
        
        val = self.load_retro_feed()
        
        if val == None:
            self.retro_feed_data = {}
        else:
            
            self.retro_feed_data= val
           
    @chronodecorator
    def save_lsi_gsim(self):


        self.dictionary = corpora.Dictionary(self.docs_preproces_gensim)
        corpus = [self.dictionary.doc2bow(text) for text in self.docs_preproces_gensim]

        print('creating model..')
        self.lsi_model = models.LsiModel(corpus, id2word=self.dictionary, num_topics= self.numTopics)
        
        with open( self.datafolder+"/lsi_model_gesim.pk",'wb') as pickle_file:
            pickle.dump(self.lsi_model, pickle_file)

        print('creating matrix ..')
        self.index = similarities.MatrixSimilarity(self.lsi_model[corpus])

        # coherence_val =compute_coh(self.dictionary,corpus,self.docs_prepoced,self.lsi_model) 
        # print("Coherence val: ",coherence_val)
        
        print('saving...matrix')
        self.index.save(self.datafolder+'/index_lsi.simil')

        self.V = gensim.matutils.corpus2dense(self.lsi_model[corpus], len(self.lsi_model.projection.s)).T / self.lsi_model.projection.s



    

    @chronodecorator
    def save_tfidf_matrix(self,):
        """En este metodo se llama si no se han construido nunca los indices 
        Siguiendo la idea del modelo vectorial se construye una matrix termino-documento
        donde los documentos son el campo page content del json que se genero cuando se escrapeo.

        A pratir de aqui se  apoya en sklearn que genera dicha matrix cuyos pesos son el tf-idf normalizado.
        Como podemos ver los documentos estan representados como vectores de pesos tfidf como plantea el modelo clasico vectorial.

        En este caso la matrix llega a tener dimensiones de  10000* 450000 y es una matri muy esparcida.
        """
        self.tf = TfidfVectorizer()
        

        print('Computing Tf-Idf Matrix...')
        self.tfidfmatrix = self.tf.fit_transform( self.docs_prepoced)
        
        print('Saving Matrix...')
        np.save(self.datafolder+'/Tf-Idf-matrix',self.tfidfmatrix) 

        print('Saving Model...')
        with open(self.datafolder+'/TfIdfVectorizer.pk', 'wb') as f:
            pickle.dump(self.tf,f)

    def load_tfidf_matrix(self):

        """Metodo para cargar la matrix de pesos del modelo ya calculada con anterioridad"""
        
        self.tfidfmatrix= np.load(self.datafolder+'/Tf-Idf-matrix.npy').all()
    
    def load_tf_vectorizer(self):
        """Metodo para cargar el modelo """
        try:
            with open(self.datafolder+'/TfIdfVectorizer.pk', 'rb') as f:
                self.tf = pickle.load(f)
            
            self.load_tfidf_matrix()

        except:
            self.save_tfidf_matrix()

    def load_svd_matrix(self):
    
        """Metodo para cargar la matrix de pesos del modelo ya calculada con anterioridad"""
        self.index = similarities.MatrixSimilarity.load(self.datafolder+'/index_lsi.simil')
    
    def load_lsi_model(self):
        """Metodo para cargar el modelo """
        
        
        try:
            with open(self.datafolder+'/lsi_model_gesim.pk', 'rb') as f:
                print("Loading model...")
                self.lsi_model = pickle.load(f)
                # print(self.lsi_model.show_topic(0  ))
                # self.lsi_model.sh
                # print('lo cargue')
                self.dictionary = corpora.Dictionary(self.docs_preproces_gensim)
                corpus = [self.dictionary.doc2bow(text) for text in self.docs_preproces_gensim]
                
                # self.dictionary = corpora.Dictionary(self.docs_preproces_gensim)
            self.load_svd_matrix()
            self.V = gensim.matutils.corpus2dense(self.lsi_model[corpus], len(self.lsi_model.projection.s)).T / self.lsi_model.projection.s




        except:
            self.save_lsi_gsim()

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



    def rank(self, query_vector,matrix):
        """ Genera el rankink de los documentos más parecidos a la query en este caso indicado por el parametro k por default los 1000
        documentos más parecidos. Se usa la similaridad de coseno ya que es la que mejor resultados ha alcansado para este modelo"""
        result = cosine_similarity(query_vector,matrix)
        
        index_sorted = np.argsort(result)
        return (result,index_sorted[0][self.count-self.k: ])

    # @chronodecorator
    def search_query(self, query, model= 'vec'):
        """Método principal del motor de búsqueda primero revisa si es un url o no. 
        Si lo es revisa si ya lo tiene indexado lo busca yllama a rank sino lo manda scrapear y con  todo el preprocesado que lleva
        y llama a rank. Esto siempre despues de haber tranforamdo el vector de consulta al espacio de los documentos indexados.


        
        """
        # k = 20
        init = time.time()
        # print(query)

        # if not self.query_opt.get(query)== None:
        
        # if model == 'lsi':
        #     return self.search_query_LSA(query)
        if model == 'lsi-gensim' or model == 'both':
            
            try:
                query_vector = self.query_opt[query]
                # print('modificado: ', query_vector)
            except:
                query = self.preprocces(query,query= True)
                query_vector = self.transformQueryGensim(query)
                # print('no la tengo')

            sims = self.index[query_vector]
            result = np.argsort(sims)[self.count-self.k:]

            pages = []
            results = []
            
            for i in result:
                pages.append((round(sims[i],4), self.file_names[i]))
                results.append(self.file_names[i])
            
            pages.reverse()
            results.reverse()





            time_took =round(time.time()-init,3)
            print('Your search took ' + str(round(time.time()-init,4))+ ' seconds')
            
            return pages,time_took


        try:
                # print('query sin proccess',query)
                # print(len(query))
                query_vector = self.query_opt[query]
                # print('modificado: ', query_vector)
        except:
                query = self.preprocces(query,query= True)
                query_vector = self.transformQery(query)
                # print('no la tengo')

        # print('Vectorizing Query')
        # query_vector = self.transformQery(query)

        result, n_Mayores = self.rank(query_vector,self.tfidfmatrix)



        pages =[ ]
        results =[]

        for i in n_Mayores:
            pages.append((round(result[0][i],4), self.file_names[i] ))
            results.append(self.file_names[i])

        pages.reverse()
        results.reverse()

    


        time_took =round(time.time()-init,3)
        print('Your search took ' + str(round(time.time()-init,3))+ ' seconds')
        
        return pages,time_took

    
    def cal_measures(self, rr,nr,ri,k):

            rr = len(rr)
            nr = len(nr)
            ri = len(ri)
            precc = precision(rr,ri)
            recb  = recall(rr,nr)
            f_med = f_medida(recb,precc)
            f1_med = f1_medida(recb,precc)
            r_prec = r_precision(k,rr )
            return precc,recb,f_med,f1_med,r_prec


    def calc_rr_nr_ri(self,query,results):
        """devuelve una tripla con
            rr: relevantes recuperados
            nr: relevantes no recuperados
            ri: irrelevantes recuperados
         """
        results = set(results)
        if not self.relevantes.get(query) == None:
            relevants = set( self.relevantes[query])
        else:
            relevants = set( self.retro_feed_data[query])
        rr = relevants.intersection(results)
        ri = results.difference(relevants)
        nr =  relevants.difference(results)

        return (rr,nr,ri)

    def preprocces(self,text,query = False):
            
        doc = re.split(r'[\W0-9]+', text.lower())

        result = ""
        for word in doc:
            # print(word)
            if word in self.stopwords:
                continue
            value =  LOOKUP.get(word)
            value_en = enL.get(word)
            if value == None and value_en == None:
                result+= word +" "
                continue
            if not value  == None:
                result += value + " "
            if not value_en == None:
                result += value_en + " "

        # result = str(result[:-1])
        # print(result)
        # a = input()


        if not query:
            self.docs_prepoced.append(result)
            result1 =result.split()
            
            # print(result1)
            self.docs_preproces_gensim.append(result1)
        else:
            return result

    def load_folder(self, folder="data/"):
        print("preprossing...")
        self.datafolder = folder
        self.file_names = []
        self.count = 0
        for file in tqdm(os.listdir(self.datafolder)):
            if file.endswith(".pdf") or file.endswith(".txt"):
                self.file_names.append(file)
                new_file = os.path.join(self.datafolder,file)
                
                if file.endswith(".txt"):
                    with open(new_file, encoding='utf-8') as fd:
                        self.preprocces(fd.read())
                        # text = fd.read()
                        self.doc_to_index[file] = self.count
                        # self.docs_prepoced.append(text)
                        # self.docs_preproces_gensim.append(text.split())

                        # if self.count %100 ==0:
                            # print(self.count)
                else:
                    self.preprocces(self.read_pdf(new_file))
                self.count+=1
    
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

            precc,recb,f_med,f1_med,r_prec = self.cal_measures( rr,nr,ri,20)

            self.generate_opt_query(key,rr,ri)

           
            self.retro_feed_data[key].append([precc,recb,f1_med,f_med,r_prec])
        else:
            # self.retro_feed_data[key] = set()
            self.retro_feed_data[key] = items

            rr , nr , ri = self.calc_rr_nr_ri(key,aux)
            # print(self.svdMatrix[ 0])
            precc,recb,f_med,f1_med,r_prec = self.cal_measures( rr,nr,ri,20)

            self.generate_opt_query(key,rr,ri)

            self.retro_feed_data[key].append([precc,recb,f1_med,f1_med,r_prec])

    def add_relevant(self,query, relevants):

        if not self.relevantes.get(query) == None:
            for item in relevants:
                if not item in self.relevantes.get(query):
                    self.relevantes[query].append(item)
        else:
            self.relevantes[query] = relevants
    def save_relevante(self):

        with open(self.datafolder+'/relevantes.json','w',encoding='utf-8') as fd:
            json.dump(self.relevantes,fd)




    def load_relevant(self):

        try :
            with open(self.datafolder+"/relevantes.json",encoding='utf-8') as fd:
                self.relevantes = json.load(fd)
                print('cargue')
            
        
        except:
            self.relevantes= {}







    def generate_opt_query(self,query,rr,ri):

        rr_len =len(rr)
        rri_len =len(ri)
        dj_r = []
        dj_ri = []

        if self.model== 'vec':
            self.V = self.tfidfmatrix
        
        for file in rr:
            dj_r.append(self.V[self.doc_to_index[file]])
            try :
                aux+= self.V[self.doc_to_index[file]]
            except:
                aux = self.V[self.doc_to_index[file]]

        for file in ri:
            dj_ri.append(self.V[self.doc_to_index[file]])
            try :
                aux_i+= self.V[self.doc_to_index[file]]
            except:
                aux_i = self.V[self.doc_to_index[file]]

        # count = 0
        # for doc in range(0,self.count):
        #     try:  
        #         result += V[doc]
        #         count+=1

        #     except:
               
        #         result = V[doc]
        #         count= 1

        # result = result - aux

        final = aux/rr_len - aux_i/rri_len
        # print(query)
        self.query_opt[query] = final
        


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
            
            with open('data/retro_feed.json','w', encoding='utf-8')as fk:
                json.dump(self.retro_feed_data,fk)

    