import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


class TOSPAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.classifier = RandomForestClassifier(random_state=42)

    def preprocess_data(self, df):
        """
        Preprocess the TOSP data and create features for analysis
        """
        # Clean description text
        df['Description'] = df['Description'].str.upper()

        # Create procedure type features
        df['procedure_type'] = df['Description'].apply(lambda x: x.split(',')[0])

        # Extract location information (e.g., BILATERAL, UNILATERAL)
        df['is_bilateral'] = df['Description'].str.contains('BILATERAL')
        df['is_unilateral'] = df['Description'].str.contains('UNILATERAL')

        # Create numerical features
        df['table_numeric'] = df['Table'].apply(lambda x:
                                                float(x[:-1]) if x[0].isdigit() else 0)

        return df

    def generate_pairs(self, df):
        """
        Generate all possible pairs of TOSP codes and their features
        """
        pairs = []
        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                code1, code2 = df.iloc[i], df.iloc[j]

                # Calculate description similarity
                desc_sim = cosine_similarity(
                    self.vectorizer.fit_transform([code1['Description'],
                                                   code2['Description']])
                )[0][1]

                # Check if procedures are related
                same_procedure = (code1['procedure_type'] ==
                                  code2['procedure_type'])

                # Check for bilateral/unilateral conflict
                bilateral_conflict = (
                        (code1['is_bilateral'] and code2['is_unilateral']) or
                        (code1['is_unilateral'] and code2['is_bilateral'])
                )

                # Calculate table difference
                table_diff = abs(code1['table_numeric'] - code2['table_numeric'])

                pairs.append({
                    'code1': code1['Code'],
                    'code2': code2['Code'],
                    'desc_similarity': desc_sim,
                    'same_procedure': same_procedure,
                    'bilateral_conflict': bilateral_conflict,
                    'table_difference': table_diff,
                    # This would be 1 for known fraudulent pairs in real data
                    'is_inappropriate': bilateral_conflict and same_procedure
                })

        return pd.DataFrame(pairs)

    def train_model(self, pairs_df):
        """
        Train a model to identify inappropriate pairs
        """
        features = ['desc_similarity', 'same_procedure', 'table_difference']
        X = pairs_df[features]
        y = pairs_df['is_inappropriate']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.classifier.fit(X_train, y_train)
        y_pred = self.classifier.predict(X_test)

        return classification_report(y_test, y_pred)

    def identify_suspicious_pairs(self, pairs_df, threshold=0.7):
        """
        Identify potentially inappropriate pairs based on model predictions
        and rules
        """
        features = ['desc_similarity', 'same_procedure', 'table_difference']
        predictions = self.classifier.predict_proba(pairs_df[features])

        pairs_df['fraud_probability'] = predictions[:, 1]
        suspicious_pairs = pairs_df[pairs_df['fraud_probability'] > threshold]

        return suspicious_pairs.sort_values('fraud_probability', ascending=False)

# Load your TOSP data
tosp_data = pd.read_csv('DataSets/CleanedDataset/combined_dataset.csv')

# Initialize analyzer
analyzer = TOSPAnalyzer()

# Preprocess data
processed_data = analyzer.preprocess_data(tosp_data)

# Generate and analyze pairs
pairs = analyzer.generate_pairs(processed_data)

# Train the model
performance_report = analyzer.train_model(pairs)

# Identify suspicious pairs
suspicious_pairs = analyzer.identify_suspicious_pairs(pairs, threshold=0.7)