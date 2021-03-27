# %matplotlib inline


from pprint import pprint
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from nltk.tokenize import RegexpTokenizer
import logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

with open("answers.txt", encoding="UTF-8") as file_in:
    docs = []
    for line in file_in:
        docs.append(line.strip())


numbers = [1, 2, 3]
squares = [x*x for x in numbers]


# Split the documents into tokens.
tokenizer = RegexpTokenizer(r'\w+')
for idx in range(len(docs)):
    docs[idx] = docs[idx].lower()  # Convert to lowercase.
    docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.

# Remove numbers, but not words that contain numbers.
docs = [[token for token in doc if not token.isnumeric()] for doc in docs]

# Remove custom stop words
stopwords = ["the", "to", "and", "that", "are", "or" "in", "of", "what", "a", "in", "it", "with",
             "at", "be", "by", "is", "on", "or", "if", "you", "can", "for", "not", "this",
             "do", "as", "they", "have", "has", "did", "an", "does", "so", "am", "but", "why",
             "how", "get", "was", "who", "had", "like",
             "me", "my", "then", "he", "we", "when", "his", "much",
             "your", "will", "just", "where", "which",
             "being", "should", "us", "their", "from",
             "because", "would", "than", "no",
             "still", "really", "her",
             "anyone", "any", "these", "him",
             "there", "many", "such", "someone",
             "here", "between", "about", "ever",
             "dont", "our", "she",
             "all", "only", "please",
             "say", "says", "else", "ve", "re", "im",
             "some", "most", "don", "during",
             "them", "been", "having", "using",
             "mean", "while", "could",
             "into", "getting", "got", "given",
             "whats", "other", "were", "very", "isn"
             "off", "since", "with", "without",
             "tell", "said", "something", "things",
             "see", "over", "off", "want", "its", "thing",
             "come", "make", "always", "going", "another", "give", "too",
             "feel", "use", "yahoo", "way", "even", "anything", "look", "also", "lot", "let",
             "die", "up", "take"]

docs = [[token for token in doc if len(token) > 1] for doc in docs]

docs = [[token for token in doc if token not in stopwords] for doc in docs]


# Remove rare and common tokens.

# Create a dictionary representation of the documents.
dictionary = Dictionary(docs)

# Filter out words that occur less than 20 documents, or more than 50% of the documents.
dictionary.filter_extremes(no_below=20, no_above=0.5)


# Bag-of-words representation of the documents.
corpus = [dictionary.doc2bow(doc) for doc in docs]


print('Number of unique tokens: %d' % len(dictionary))
print('Number of documents: %d' % len(corpus))


# Train LDA model.

# Set training parameters.
#num_topics = 12
num_topics = 30
chunksize = 2000
passes = 20
iterations = 400
eval_every = None  # Don't evaluate model perplexity, takes too much time.

# Make a index to word dictionary.
temp = dictionary[0]  # This is only to "load" the dictionary.
id2word = dictionary.id2token

model = LdaModel(
    corpus=corpus,
    id2word=id2word,
    chunksize=chunksize,
    alpha='auto',
    eta='auto',
    iterations=iterations,
    num_topics=num_topics,
    passes=passes,
    eval_every=eval_every
)


top_topics = model.top_topics(corpus)  # , num_words=20)

# Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
print('Average topic coherence: %.4f.' % avg_topic_coherence)

pprint(top_topics)


with open('LDA.txt', 'w') as f:
    for topic in top_topics:
        for entry in topic:
            f.write(str(entry))
            f.write("\n")


import pyLDAvis
import pyLDAvis.gensim as gensimvis


vis_data1 = gensimvis.prepare(model, corpus, dictionary, sort_topics=False)


pyLDAvis.display(vis_data1)
pyLDAvis.save_html(vis_data1, 'lda.html')