import pandas as pd

resenha = pd.read_csv("files/imdb-reviews-pt-br.csv")
resenha.head()

resenha

from sklearn.model_selection import train_test_split

treino, teste, classse_treino, classe_teste = train_test_split(resenha.text_pt, 
                                                 resenha.sentiment, 
                                                 random_state=42)

from sklearn.linear_model import LogisticRegression

regressao_logistica = LogisticRegression()
regressao_logistica.fit(treino, teste)
acuracia = regressao_logistica.score(teste, classe_teste)

#Analise do texto
print(resenha.sentiment.value_counts())

classificacao = resenha.sentiment.replace(["neg", "pos"], [0,1])
resenha["classsificacao"] = classificacao

resenha.head()

from sklearn.feature_extraction.text import CountVectorizer

texto = ["Assistir um filme otimo", "Assiti um filme ruim"]

vetorizar = CountVectorizer()
bag_of_words = vetorizar.fit_transform(texto)

vetorizar.get_feature_names_out()

sparsa = pd.DataFrame.sparse.from_spmatrix(bag_of_words, columns=vetorizar.get_feature_names_out())

sparsa

#Dados do IMBD
def classificar_texto(texto, coluna_texto, coluna_classificacao):
    vetorizar = CountVectorizer(max_features=50)
    bag_of_words = vetorizar.fit_transform(texto[coluna_texto])
    
    treino, teste, classse_treino, classe_teste = train_test_split(bag_of_words, 
                                                    texto[coluna_classificacao], 
                                                    random_state=42)


    regressao_logistica = LogisticRegression()
    regressao_logistica.fit(treino, classse_treino)
    return regressao_logistica.score(teste, classe_teste)
print(classificar_texto(resenha, "text_pt", "classsificacao"))

#Instalando WordCloud
#pip install wordcloud


from wordcloud import WordCloud

#Criar uma variavel com todas as valavras
todas_palavras = ''.join([texto for texto in resenha["text_pt"]])

nuvem_palavras = WordCloud(width= 800, height=500, max_font_size=110, collocations=False).generate(todas_palavras)

import matplotlib.pyplot as plt

plt.figure(figsize=(10,7))
plt.imshow(nuvem_palavras, interpolation='bilinear')
plt.axis("off")
plt.show()

#Segregando os sentimentos
resenha.query("sentiment == 'pos'")

def nuvem_palavras_neg(texto, coluna_texto):
    texto_negativo = texto.query("sentiment == 'neg'")
    todas_palavras = ''.join([texto for texto in texto_negativo[coluna_texto]])

    nuvem_palavras = WordCloud(width= 800, height=500, max_font_size=110, collocations=False).generate(todas_palavras)

    plt.figure(figsize=(10,7))
    plt.imshow(nuvem_palavras, interpolation='bilinear')
    plt.axis("off")
    plt.show()

def nuvem_palavras_pos(texto, coluna_texto):
    texto_positivo = texto.query("sentiment == 'pos'")
    todas_palavras = ''.join([texto for texto in texto_positivo[coluna_texto]])

    nuvem_palavras = WordCloud(width= 800, height=500, max_font_size=110, collocations=False).generate(todas_palavras)

    plt.figure(figsize=(10,7))
    plt.imshow(nuvem_palavras, interpolation='bilinear')
    plt.axis("off")
    plt.show()

nuvem_palavras_neg(resenha, "text_pt")
nuvem_palavras_pos(resenha, "text_pt")

#NLTK
import nltk
a = ["um filme ruim", "um filme bom"]
frequencia = nltk.FreqDist()

from nltk import tokenize

frase = "Bem vindo ao mundo do PLN"
token_espaco = tokenize.WhitespaceTokenizer()
token_frase = token_espaco.tokenize(frase)
print(token_frase)

from nltk.tokenize import WordPunctTokenizer
token_pontuacao = WordPunctTokenizer()

import seaborn as sns
def pareto(df, coluna_texto, quantidade):
    todas_palavras = ' '.join(df[coluna_texto])  # Concatenar todas as frases em uma única string
    token_frase = token_pontuacao.tokenize(todas_palavras)  # Tokenizar as frases
    frequencia = nltk.FreqDist(token_frase)  # Calcular a frequência das palavras
    df_frequencia = pd.DataFrame({"Palavra": list(frequencia.keys()), "Frequencia": list(frequencia.values())})
    df_frequencia = df_frequencia.nlargest(columns="Frequencia", n=quantidade)  # Selecionar as mais frequentes
    plt.figure(figsize=(12,8))
    ax = sns.barplot(data=df_frequencia, x="Palavra", y="Frequencia", color='gray')
    ax.set(ylabel="Contagem")
    plt.show()

pareto(resenha, "text_pt", 10)

#STOP WORD - PALAVRAS IRRELEVANTES
palavras_irrelevantes = nltk.corpus.stopwords.words("portuguese")
print(palavras_irrelevantes)
frase_prcessada = list()
for opniao in resenha.text_pt:
    nova_frase = list()
    palavras_texto = token_espaco.tokenize(opniao)
    for palavra in palavras_texto:
        if palavra not in palavras_irrelevantes:
            nova_frase.append(palavra)
    frase_prcessada.append(' '.join(nova_frase))

resenha["tratamento_1"] = frase_prcessada
resenha.head()

pareto(resenha, "tratamento_1", 10)

from nltk import tokenize
frase = "Olá mundo!"

#TRATAMENTO COM PONTUAÇÃO
from string import punctuation
pontuacao = list()
for ponto in punctuation:
    pontuacao.append(ponto)

pontuacao_stopwords = pontuacao + palavras_irrelevantes

frase_processada = list()
for opiniao in resenha["tratamento_1"]:
    nova_frase = list()
    palavras_texto = token_pontuacao.tokenize(opiniao)
    for palavra in palavras_texto:
        if palavra not in pontuacao_stopwords:
            nova_frase.append(palavra)
    frase_processada.append(' '.join(nova_frase))

resenha["tratamento_2"] = frase_processada
pareto(resenha, "tratamento_2", 10)

#TRATAMENTO SEM ACENTO
from unidecode import unidecode

acentos = "ótimos péssimo não é tão"
teste = unidecode(acentos)
print(teste)

sem_acentos = [unidecode(texto) for texto in resenha["tratamento_2"]]

stop_words_sem_acento = [unidecode(texto) for texto in pontuacao_stopwords]

resenha["tratamento_3"] = sem_acentos

frase_processada = list()
for opiniao in resenha["tratamento_3"]:
    nova_frase = list()
    palavras_texto = token_pontuacao.tokenize(opiniao)
    for palavra in palavras_texto:
        if palavra not in pontuacao_stopwords:
            nova_frase.append(palavra)
    frase_processada.append(' '.join(nova_frase))

resenha["tratamento_3"] = frase_processada
resenha.head()


frase_processada = list()
for opiniao in resenha["tratamento_3"]:
    nova_frase = list()
    opniao = opniao.lower()
    palavras_texto = token_pontuacao.tokenize(opiniao)
    for palavra in palavras_texto:
        if palavra not in stop_words_sem_acento:
            nova_frase.append(palavra)
    frase_processada.append(' '.join(nova_frase))
resenha["tratamento_4"] = frase_processada


stemmer = nltk.RSLPStemmer()
stemmer.stem("corredor")

stopwords_sem_acento = [unidecode(palavra) for palavra in palavras_irrelevantes]

frase_processada = list()
for opiniao in resenha["tratamento_4"]:
    nova_frase = list()
    palavras_texto = token_pontuacao.tokenize(opiniao)
    for palavra in palavras_texto:
        if palavra not in stopwords_sem_acento:
            nova_frase.append(stemmer.stem(palavra))
    frase_processada.append(' '.join(nova_frase))

resenha["tratamento_5"] = frase_processada

nuvem_palavras_neg(resenha, "tratamento_5")


pareto(resenha, "tratamento_5", 10)

from nltk import ngrams

frase = "Assisti um ótimo filme."
frase_separada = token_espaco.tokenize(frase)
pares = ngrams(frase_separada, 2)
list(pares)

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(lowercase=False, ngram_range = (1,2))
vetor_tfidf = tfidf.fit_transform(resenha["tratamento_5"])
treino, teste, classe_treino, classe_teste = train_test_split(vetor_tfidf, resenha["classificacao"], random_state = 42)
regressao_logistica.fit(treino, classe_treino)
acuracia_tfidf_ngrams = regressao_logistica.score(teste, classe_teste)
print(acuracia_tfidf_ngrams)

tfidf = TfidfVectorizer(lowercase=False)
vetor_tfidf = tfidf.fit_transform(resenha["tratamento_5"])
treino, teste, classe_treino, classe_teste = train_test_split(vetor_tfidf, resenha["classificacao"], random_state = 42)
regressao_logistica.fit(treino, classe_treino)
acuracia_tfidf = regressao_logistica.score(teste, classe_teste)
print(acuracia_tfidf)

pesos = pd.DataFrame(
    regressao_logistica.coef_[0].T,
    index = tfidf.get_feature_names()
)
pesos.nlargest(10, 0)