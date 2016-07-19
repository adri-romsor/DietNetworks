import numpy
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from HTMLParser import HTMLParser

class ReutersParser(HTMLParser):
    
    def __init__(self):
        self.in_article = False
        self.arianne_thread = []
        self.current_article = {}
        self.parsed_articles = []
        HTMLParser.__init__(self)
    
    def handle_starttag(self, tag, attrs):
        self.arianne_thread.append(tag)
        
        if tag == "reuters":
            # Beginning of a new article
            self.in_article = True
            self.current_article = {}

            for (att, val) in attrs:
                if att != "topics":
                    self.current_article[att] = val

    def handle_endtag(self, tag):
        assert tag == self.arianne_thread[-1]
        del self.arianne_thread[-1]
        
        if tag == "reuters":
            # End of an article
            self.in_article = False
            self.parsed_articles.append(self.current_article)
            
    def handle_data(self, data):
        
        if self.in_article:
            
            if self.arianne_thread[-1] == "d":
                
                tag = self.arianne_thread[-2]
                
                if tag not in self.current_article.keys():
                    self.current_article[tag] = []
                self.current_article[tag].append(data)
            
            else:
            
                tag = self.arianne_thread[-1]
                
                if tag not in self.current_article.keys():
                    self.current_article[tag] = data
                else:
                    self.current_article[tag] += data
        
    def get_parsed_data(self):
        return self.parsed_articles


class ReutersDataset(object):
    
    def __init__(self, folder="/data/lisa/data/reuters/"):
        self.folder = folder
        
    def load_data(self):
        
        text_data = self.read_text_data()
        exchanges, orgs, people, places, topics = self.read_word_lists()
        
        # Preprocess the texts to a standard format
        texts = []
        for t in text_data:
            
            full_text = ""
            if 'title' in t:
                full_text += t['title'] + " "
            if 'body' in t:
                full_text += t['body']
            
            full_text_no_punctuation = re.sub("[^a-zA-Z0-9]", " ", full_text)
            
            words = full_text_no_punctuation.lower().split()
            texts.append(" ".join(words))
            
        # Obtain a bag of words representation for the texts
        vectorizer = CountVectorizer(analyzer="word", tokenizer=None,
                                     preprocessor=None, stop_words="english")
        bag_of_words = vectorizer.fit_transform(texts).toarray()
        
        # Perform TF-IDF preprocessing on the bag of words
        transformer = TfidfTransformer()
        bag_of_words = transformer.fit_transform(bag_of_words).toarray()
        bag_of_words = bag_of_words.astype("float32")

        # Produce targets for every text
        classes = exchanges + orgs + people + places + topics
        targets = numpy.zeros((len(texts), len(classes)), dtype="float32")
        
        for idx, t in enumerate(text_data):
            
            y_exchanges = [0.0 for e in exchanges]
            if 'exchanges' in t:
                for e in t['exchanges']:
                    y_exchanges[exchanges.index(e)] = 1.0
        
            y_orgs = [0.0 for e in orgs]
            if 'orgs' in t:
                for e in t['orgs']:
                    y_orgs[orgs.index(e)] = 1.0
                    
            y_people = [0.0 for e in people]
            if 'people' in t:
                for e in t['people']:
                    y_people[people.index(e)] = 1.0
                    
            y_places = [0.0 for e in places]
            if 'places' in t:
                for e in t['places']:
                    y_places[places.index(e)] = 1.0
                    
            y_topics = [0.0 for e in topics]
            if 'topics' in t:
                for e in t['topics']:
                    y_topics[topics.index(e)] = 1.0
                    
            y = y_exchanges + y_orgs + y_people + y_places + y_topics
            y = numpy.array(y).astype("float32")
            targets[idx] = y


        # Split the dataset into train and test
        train_indices = []
        test_indices = []
        
        for idx, t in enumerate(text_data):
            split =  t['lewissplit']
            if split == "TRAIN":
                train_indices.append(idx)
            elif split == "TEST":
                test_indices.append(idx)
                
        train_x = bag_of_words[train_indices]
        train_y = targets[train_indices]
        test_x = bag_of_words[test_indices]
        test_y = targets[test_indices]
        
        return (train_x, train_y), (test_x, test_y)
        
    def read_text_data(self):
        nb_files = 22
        filenames_format = self.folder + "reut2-%03i.sgm"
        
        parser = ReutersParser()
        for file_idx in range(nb_files):
            filename = filenames_format % file_idx
            
            with open(filename, 'r') as f:
                parser.feed(f.read())
                
        return parser.get_parsed_data()
        
    def read_word_lists(self):
        """
        Reads the word lists associated with the dataset and returns them as
        a tuple of lists of strings. The lists are, respectively, associated
        with the "exchanges", "orgs", "people", "places" and "topics"
        categories.
        """
        return (self._read_txt_list(self.folder + "all-exchanges-strings.lc.txt"),
                self._read_txt_list(self.folder + "all-orgs-strings.lc.txt"),
                self._read_txt_list(self.folder + "all-people-strings.lc.txt"),
                self._read_txt_list(self.folder + "all-places-strings.lc.txt"),
                self._read_txt_list(self.folder + "all-topics-strings.lc.txt"))
    
    def _parse_sgm_file(self, filename):
        with open(filename, 'r') as f:
            result = None
        return result
        
    def _read_txt_list(self, filename):
        with open(filename, 'r') as f:
            lines = [l.strip() for l in f.readlines() if len(l.strip()) > 0]
        return lines

ReutersDataset().load_data()