import time
start = time.time()
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import re
import numpy as np
import pandas as pd
from pprint import pprint
import nltk
nltk.download('stopwords')
import spacy
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
import os
import sys
import numpy as np
from scipy import spatial

# # Corpus docs
corpus = '''Data science is growing up fast. Over the past five years companies have invested billions to get the most-talented data scientists to set up shop, amass zettabytes of material, and run it through their deduction machines to find signals in the unfathomable volume of noise. It’s working—to a point. Data has begun to change our relationship to fields as varied as language translation, retail, health care, and basketball.

But despite the success stories, many companies  getting the value they could from data science. Even well-run operations that generate strong analysis fail to capitalize on their insights. Efforts fall short in the last mile, when it comes time to explain the stuff to decision makers.

In a question on Kaggle’s 2017 survey of data scientists, to which more than 7,000 people responded, four of the top seven “barriers faced at work” were related to last-mile issues, not technical ones: “lack of management/financial support,” “lack of clear questions to answer,” “results not used by decision makers,” and “explaining data science to others.” Those results are consistent with what the data scientist Hugo Bowne-Anderson found interviewing 35 data scientists for his podcast; as he wrote in a 2018 HBR.org article, “The vast majority of my guests tell [me] that the key skills for data scientists are the abilities to learn on the fly and to communicate well in order to answer business questions, explaining complex results to nontechnical stakeholders.

In my work lecturing and consulting with large organizations on data visualization (dataviz) and persuasive presentations, I hear both data scientists and executives vent their frustration. Data teams know they’re sitting on valuable insights but can’t sell them. They say decision makers misunderstand or oversimplify their analysis and expect them to do magic, to provide the right answers to all their questions. Executives, meanwhile, complain about how much money they invest in data science operations that don’t provide the guidance they hoped for. They don’t see tangible results because the results are communicated in their language.'''

# Convert corpus to docs
data = list(corpus.split("."))

# Handle emails, newline,quotes etc
data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]
data = [re.sub('\s+', ' ', sent) for sent in data]
data = [re.sub("\'", "", sent) for sent in data]

# Tokeinizing the sentence using gensim preprocess method
def sent_tokenize(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
data_words = list(sent_tokenize(data))

# Creating bigram and trigram for better predictions
bigram = gensim.models.Phrases(
    data_words, min_count=5,
    threshold=100)

trigram = gensim.models.Phrases(bigram[data_words],
                                threshold=100)

bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def bigrams_create(texts):
    return [bigram_mod[doc] for doc in texts]

def trigrams_create(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

# lemmatization using spacy module
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

data_words_nostops = remove_stopwords(data_words)
data_words_bigrams = bigrams_create(data_words_nostops)
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
data_lemmatized = lemmatization(data_words_bigrams,
                                allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']
                                )


id2word = corpora.Dictionary(data_lemmatized)
texts = data_lemmatized
corpus = [id2word.doc2bow(text) for text in texts]   # creating sparse bag of words

for topic_num in range(2,10):
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=topic_num,
                                               random_state=100,
                                               update_every=1,
                                               # chunksize=100,
                                               # passes=10,
                                               alpha='auto',
                                               per_word_topics=True)

    # pprint(lda_model.print_topics())
    # doc_lda = lda_model[corpus]
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='u_mass')
    coherence_lda = coherence_model_lda.get_coherence()

    # print(f'topic_num : {topic_num} and coherence Score: {coherence_lda}')
    if topic_num == 2:
        temp = 0
    else:
        temp = coherence_lda

    if coherence_lda < temp:
        best_coherent_score = coherence_lda
        best_topic_num = topic_num

# print(f"Best Score is {best_coherent_score} for topic number : {best_topic_num}")

def getBestModelTopics(best_topic_num):
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=best_topic_num,
                                                random_state=100,
                                                update_every=1,
                                                # chunksize=100,
                                                # passes=10,
                                                alpha='auto',
                                                per_word_topics=True)
    return lda_model.print_topics()

result = getBestModelTopics(best_topic_num)

#creating input format ready for glove model
getKeywords = []
for i in range(0, best_topic_num):
    temp_sent = result[i][1]
    filter_temp_sent = re.sub(r'[^A-Za-z]+', ' ', temp_sent).strip()
    each_topic_list = list(filter_temp_sent.split(" "))
    getKeywords.append(each_topic_list)

# using glove model finding comparision between keywords

embeddings_dict = {}
glove_path = os.path.join(os.getcwd(), "glove.6B.50d.txt")
with open(glove_path, 'r', encoding = 'utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector

predefined_category = "technology"  # give in lowercase as per glove
predefined_embed = embeddings_dict[predefined_category]
dominant_topic = []
for key, val in enumerate(getKeywords):

    res = 0
    for inner_key, keyval in enumerate(val):
        res =+ 1 - spatial.distance.cosine(predefined_embed, embeddings_dict[keyval])
    dominant_topic.append(res)


# print(dominant_topic)
index_max = np.argmax(dominant_topic)

# Result :
print(f"topic no: {index_max} is dominating topic for given category : {predefined_category}")
print(f"Spatial distance between category and best suited topic found: {dominant_topic[index_max]}")
print(f"Best Topic Keyword found is : ")
pprint({result[index_max][1]})

