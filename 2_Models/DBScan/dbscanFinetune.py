import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from string import punctuation
import nltk
import re
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
from nltk.stem import WordNetLemmatizer


class TOSPAnalyzer:
    def __init__(self):
        self.encode = LabelEncoder()
        self.lemmatizer = WordNetLemmatizer()
        self.vectorize = TfidfVectorizer(stop_words='english')
        self.classifier = DBSCAN(eps=0.29, min_samples=1, metric='cosine') 
        
    def remove_stopwords(self, description):
        # nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
        
        if isinstance(description, str):
            return ' '.join([word for word in description.split() if word.lower() not in stop_words])
        return description 
    
    def remove_punctuation(self, description):
        for word in description:
            if word in list(punctuation):
                description = description.replace(word, ' ')
        
        return description
    
    def lemmatize_words(self, description):
        # Split the description into words and lemmatize each word
        lemmatized_description = ' '.join([self.lemmatizer.lemmatize(word) for word in description.split()])
        return lemmatized_description
        
    def trim_spaces(self, description):
        return re.sub(r'\s+', ' ', description).strip()
        

    def preprocess_data(self, df):
        
        df.drop(columns=['S/N', 'Classification'], inplace=True)
        
        # Encode Table column.
        # self.encode.fit_transform(df["Table"])
        
        # **Remove stop words first.**
        df['Description'] = df['Description'].apply(self.remove_stopwords)
        
        # Trim spaces after removing punctuation.
        df['Description'] = df['Description'].apply(self.remove_punctuation)
        df['Description'] = df['Description'].apply(self.trim_spaces)
        
        df['Description'] = df['Description'].apply(self.lemmatize_words)
        
        # Capitalize.
        df['Description'] = df['Description'].str.lower()
        
        
        return df
    
    def run_analyzer(self, processed_description):
        bow_embeddings = self.vectorize.fit_transform(processed_description).toarray()
        prediction = self.classifier.fit_predict(bow_embeddings)
        
        return prediction
    

df = pd.read_csv("DataSets/CleanedDataset/SD_Cardiovascular.csv")

sentences = df['Description'].to_list()
df['original_description'] = df['Description'].copy()

analyzer = TOSPAnalyzer()
processed_data = analyzer.preprocess_data(df)

predicted_labels = analyzer.run_analyzer(processed_data['Description'])
df["Labels"] = predicted_labels

print("Cluster labels:", predicted_labels)

# df['Description'] = df['original_description']
# df.drop(columns=['original_description'], inplace=True)
        
df_sorted = df.sort_values(by='Labels', ascending=True)

df_sorted.to_csv("out_dbscan.csv", index=False)


