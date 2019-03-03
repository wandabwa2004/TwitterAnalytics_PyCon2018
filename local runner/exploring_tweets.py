import pandas as pd
from exploration.word2vec_and_LDA import word_count, word2vec_model, \
    bag_of_words_corpus, gensim_dict, tfidf_corpus, tsne_model_fit, \
    plotly_tsne, plot_tsne, lda_model


perplexity = [10, 50, 75, 100]
iterations = [1000, 5000]
no_below = 15
no_above = 0.5
keep_n = 100000
top = 10


tweet_df = pd.read_csv('snakebite.csv', sep=',', encoding='latin-1')
top_words = word_count(tweet_df, top)
print(top_words)

keyword_list = ['snakebite', 'snake', 'anti', 'venom', 'n', 'bite']
sentence_dictionary = gensim_dict(tweet_df, keyword_list)
sentence_list = sentence_dictionary['sentence_list']
dictionary = sentence_dictionary['dictionary']

bow_corpus = bag_of_words_corpus(sentence_list,
                                 dictionary, no_below, no_above, keep_n)
corpus_tfidf = tfidf_corpus(bow_corpus)
lda_topics = lda_model(corpus_tfidf, dictionary)
word2vec_corpus = word2vec_model(sentence_list)
vocab = word2vec_corpus['vocab']
X_transformed = word2vec_corpus['X_transformed']

df = tsne_model_fit(X_transformed, vocab, perplexity, iterations)
plot_tsne(df['x'], df['y'], df['word'])
plotly_tsne(df)
