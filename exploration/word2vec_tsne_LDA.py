import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import Counter 
import ast
from gensim.models import Word2Vec, ldamodel
from gensim import models
from pprint import pprint
import pyLDAvis.gensim
import plotly
from plotly.graph_objs import Scatter, Layout

tweet_frame = pd.read_csv('snakebite.csv', sep=',', encoding='latin-1')

word_list = [] 
sentences = tweet_frame['cleaned'].tolist()
for i in range(tweet_frame['cleaned'].shape[0]):
    sentence = sentences[i]
    sentence = ast.literal_eval(sentence)
    word_list = word_list + sentence

count_all = Counter()
count_all.update(word_list)
count_all.most_common(10)    

keywords = ['snakebite','snake','anti', 'venom', 'n', 'bite']

sentence_list=[] 
for i in range(tweet_frame['cleaned'].shape[0]):
    l = ast.literal_eval(tweet_frame['cleaned'][i])
    resultwords  = [word for word in l if word.lower() not in keywords]
    sentence_list.append(resultwords) 

dictionary = gensim.corpora.Dictionary(sentence_list)
count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break


dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

bow_corpus = [dictionary.doc2bow(doc) for doc in sentence_list]
bow_corpus[100]

tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]
for doc in corpus_tfidf:
    pprint(doc)
    break

lda_tfidf = ldamodel.LdaModel(corpus_tfidf, num_topics=10)
lda_tfidf.save('lda_model')

#model =  ldamodel.LdaModel.load('lda_model')

topics = lda_tfidf.show_topics(num_topics=10, num_words=10, log=False, formatted=False)

words = dictionary.token2id  # saving a dictionary of word and id 

topics_decoded = dict()

for i in range(len(topics)):
        topic_no = 'Topic ' + str(i)
        topics_decoded[topic_no] = {}
        v = topics[i][1]
        for j in range(len(v)):
            word = list(words.keys())[(list(words.values())).index(int(v[j][0]))]
            topics_decoded[topic_no][word] = v[j][1]
            

pyLDAvis.enable_notebook()
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)     
pyLDAvis.display(pyLDAvis.gensim.prepare(lda_tfidf, corpus_tfidf, dictionary))

model_CBOW = Word2Vec(sentence_list, size=300, window= 2, min_count =5, workers=1)
model_SG = Word2Vec(sentence_list, size=300, window= 5, min_count =5, workers=1, sg = 1)
#model_doc2vec = gensim.models.Doc2Vec(size=300, window=10, min_count=5, workers=11,alpha=0.025, min_alpha=0.025) 
model_CBOW.wv.most_similar('ntd', topn = 20)
model_SG.wv.most_similar('ntd', topn = 20)

vocab = list(model_CBOW.wv.vocab)
X_CBOW = model_CBOW[vocab]

perplexity = [10, 50, 75, 100]
iterations = [1000,5000]
results = pd.DataFrame(columns = ['Perplexity', 'Iterations', 'Divergence'])
for i in range(len(perplexity)):
    for j in range(len(iterations)):
        tsne = TSNE(n_components=2, random_state =0, perplexity  = perplexity[i], learning_rate=200, n_iter = iterations[j], verbose = 2)
        X_tsne = tsne.fit_transform(X_CBOW)
        results = results.append({'Perplexity': perplexity[i], 'Iterations': iterations[j], 'Divergence': tsne.kl_divergence_}, ignore_index = True)
        print('')
        print('')
        print('')
        print('The KL divergence for perplexity: %d and iterations: %d is %.4f' %(perplexity[i], iterations[j], tsne.kl_divergence_))
        print('')
        print('')
        print('')

tsne = TSNE(n_components=2, perplexity=results[results['Divergence'] == min(results['Divergence'])]
['Perplexity'].values[0], learning_rate=200, n_iter=int(results[results['Divergence'] == min(results['Divergence'])]
                                                                                    ['Iterations'].values[0]), verbose=2, random_state=42)

   
X_tsne = tsne.fit_transform(X_CBOW)

df = pd.concat([pd.DataFrame(X_tsne),
                pd.Series(vocab)],
               axis=1)

df.columns = ['x', 'y', 'word']


def plot_tsne(x, y, word):
    plt.figure(figsize=(15, 10))
    plt.margins(0)
    plt.axis('on')
    plt.title('Visualising Word2Vec embeddings using t-sne')    
    for i, txt in enumerate(word):
        plt.annotate(txt, (x.iloc[i], y.iloc[i]))
    plt.scatter(x, y, color = 'g', cmap='hsv', alpha=0.9, marker='x', s=0.6, lw=1, edgecolor='') # don't use edges
    plt.show()


plot_tsne(df['x'], df['y'], df['word'])


trace = Scatter(
    x = df['x'],
    y = df['y'],
    #mode = 'markers+text',  # uncomment if you want markers along with text
    mode = 'text',
    text = df['word'],
    textposition='bottom',
    marker = dict(
        color = '#FFBAD2',
        line = dict(width = 1)
    )
)
data = [trace]

plotly.offline.plot({
    "data": data,
    "layout": Layout(title="'Visualising Word2Vec Embeddings using t-SNE'")
})

