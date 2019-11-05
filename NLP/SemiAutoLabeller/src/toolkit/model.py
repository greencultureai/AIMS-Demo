from .autolabel import Preprocessor, AutoLabeller, check_labels
from .autolabel import recommend_words
from sklearn.naive_bayes import MultinomialNB

class MLModel():
    def __init__(self):
        self.stopwords_path = "data/stopwords.csv"
        return
    
    def run(self, data, label):
        corpus = data['content']

        # Initialise Labels
        label = check_labels(data, label)

        # Text Preprocessing
        preprocessor = Preprocessor()
        preprocessed_corpus = preprocessor.corpus_preprocess(corpus=corpus, stopwords_path=self.stopwords_path)
        preprocessor.corpus_replace_bigrams(corpus=preprocessed_corpus, min_df=50, max_df=500)
        data['content'] = preprocessor.corpus_replace_bigrams(corpus=preprocessed_corpus, min_df=50, max_df=500)

        # Enrich Labels
        autoLabeller = AutoLabeller(label, corpus, data)
        enriched_label = autoLabeller.train()

        # Predict results
        mnb = MultinomialNB()
        ypred = autoLabeller.apply(mnb, "content")
        labelled_data = data[["content"]].join(ypred)
        
        return enriched_label, labelled_data