

#RR relevantes recuperados
#RI recuperados irrelevantes
#NR relevantes no recuperados


def precision (RR, RI):
    if (RR+RI) == 0:
        return 0
    return (RR/(RR+RI))*100

def recall(RR,NR):
    if (RR+NR) == 0:
        return 0
    return (RR/(RR+NR))*100

def f1_medida(r,p):
    if r==0 or p==0:
        return 0
    return 2 *(p*r)/(p+r)

def f_medida(r,p,b = 1):
    return ((1+(b^2))*p*r)/ (b^2)*p + r

#aqui son los primeros r documentos del ranking, de hay cuantos son recuperados relevantes
def r_precision(r, RR):
    return (RR/ r) *100
