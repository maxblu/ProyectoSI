from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle
import json
import time
import numpy as np
from pprint import pprint
from tqdm import tqdm
from preprocces import Lematize

# from gensim.summarization import summarize



class CharlotteSearchEngine():
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
        self.total_count= 0 
        self.scraped_docs = []
        self.sumary = {}
        self.scraped_sites_indexed = { }
        self.scraped_sites= []
        self.lematizer = Lematize()

        print('Loading Docs...')
        self.load_json_docs()

        if os.path.exists('data/TfIdfVectorizer.pk'):
            self.load_tf_vectorizer()
            self.load_tfidf_matrix()
        else:
            self.tf = TfidfVectorizer()
            self.save_tfidf_matrix()

        
    def load_json_docs(self):
        """Método para cargar los sitios scrapeados y construir el indice de ser necesario"""
        
        i = 0
        "Wikipedia"
        with open('data/Wikipedia.jsonl',encoding='utf-8') as f:
            for line in tqdm(f):
                data = json.loads(line)
                self.scraped_docs.append(data['page content'])
                self.scraped_sites.append(data['page'])
                self.scraped_sites_indexed[data['page']] = i

                # self.sumary = summarize( data['page content'],ratio= 0.1)
                i+=1

    def save_tfidf_matrix(self,):
        """En este metodo se llama si no se han construido nunca los indices 
        Siguiendo la idea del modelo vectorial se construye una matrix termino-documento
        donde los documentos son el campo page content del json que se genero cuando se escrapeo.

        A pratir de aqui se  apoya en sklearn que genera dicha matrix cuyos pesos son el tf-idf normalizado.
        Como podemos ver los documentos estan representados como vectores de pesos tfidf como plantea el modelo clasico vectorial.

        En este caso la matrix llega a tener dimensiones de  10000* 450000 y es una matri muy esparcida.
        """

        print('Computing Tf-Idf Matrix...')
        self.tfidfmatrix = self.tf.fit_transform( self.scraped_docs)
        
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



    def rank(self, query_vector,k=10):
        """ Genera el rankink de los documentos más parecidos a la query en este caso indicado por el parametro k por default los 1000
        documentos más parecidos. Se usa la similaridad de coseno ya que es la que mejor resultados ha alcansado para este modelo"""
        result = self.cosinesimilarity(query_vector)
        
        index_sorted = np.argsort(result)
        return (result,index_sorted[0][self.total_count-k: ])

    def search_query(self, query,weburl= False):
        """Método principal del motor de búsqueda primero revisa si es un url o no. 
        Si lo es revisa si ya lo tiene indexado lo busca yllama a rank sino lo manda scrapear y con  todo el preprocesado que lleva
        y llama a rank. Esto siempre despues de haber tranforamdo el vector de consulta al espacio de los documentos indexados.


        
        """
        init = time.time()

        print('Vectorizing Query')
        query_vector = self.transformQery(query)

        result, n_Mayores = self.rank(query_vector)

        pages =[ ]

        for i in n_Mayores:
            pages.append((round(result[0][i],3),self.scraped_sites[i]))

        pages.reverse()

        time_took =round(time.time()-init,3)
        print('Your search took ' + str(round(time.time()-init,3))+ ' seconds')
        
        return pages,time_took

# if __name__ == "__main__":