import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns


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

    def print_analysis_results(self, df, pairs_df=None, suspicious_pairs=None):
        """
        Print comprehensive analysis results
        """
        print("\n=== TOSP Analysis Results ===\n")

        # 1. Basic Statistics
        print("1. Basic Statistics:")
        print(f"Total number of procedures: {len(df)}")
        print(f"Number of unique procedure types: {df['procedure_type'].nunique()}")
        print(f"Number of tables: {df['Table'].nunique()}")
        print("\n" + "=" * 50 + "\n")

        # 2. Procedure Type Distribution
        print("2. Procedure Type Distribution:")
        proc_dist = df['procedure_type'].value_counts()
        for proc, count in proc_dist.items():
            print(f"{proc}: {count}")
        print("\n" + "=" * 50 + "\n")

        # 3. Table Distribution
        print("3. Table Distribution:")
        table_dist = df['Table'].value_counts()
        for table, count in table_dist.items():
            print(f"Table {table}: {count}")
        print("\n" + "=" * 50 + "\n")

        # 4. Bilateral/Unilateral Procedures
        print("4. Bilateral/Unilateral Procedures:")
        bilateral = df[df['is_bilateral']]
        unilateral = df[df['is_unilateral']]
        print(f"Bilateral procedures: {len(bilateral)}")
        print(f"Unilateral procedures: {len(unilateral)}")
        print("\nBilateral Procedures:")
        for _, row in bilateral.iterrows():
            print(f"- {row['Code']}: {row['Description']}")
        print("\nUnilateral Procedures:")
        for _, row in unilateral.iterrows():
            print(f"- {row['Code']}: {row['Description']}")
        print("\n" + "=" * 50 + "\n")

        # 5. Potential Conflicts
        if pairs_df is not None:
            print("5. Code Pair Analysis:")
            print(f"Total number of possible pairs: {len(pairs_df)}")
            print(f"Pairs with high similarity (>0.8): {len(pairs_df[pairs_df['desc_similarity'] > 0.8])}")
            print(f"Pairs with same procedure type: {len(pairs_df[pairs_df['same_procedure']])}")
            print(f"Pairs with bilateral/unilateral conflict: {len(pairs_df[pairs_df['bilateral_conflict']])}")
            print("\n" + "=" * 50 + "\n")

        # 6. Suspicious Pairs
        if suspicious_pairs is not None:
            print("6. Suspicious Pairs:")
            print("\nTop potentially inappropriate pairs:")
            for _, row in suspicious_pairs.head().iterrows():
                print(f"\nPair: {row['code1']} and {row['code2']}")
                print(f"Fraud Probability: {row['fraud_probability']:.2f}")
                print(f"Description Similarity: {row['desc_similarity']:.2f}")
                print(f"Same Procedure Type: {'Yes' if row['same_procedure'] else 'No'}")
                print(f"Bilateral/Unilateral Conflict: {'Yes' if row['bilateral_conflict'] else 'No'}")
            print("\n" + "=" * 50 + "\n")


# Example usage:
if __name__ == "__main__":
    # Load data
    data = pd.read_csv('DataSets/CleanedDataset/SL_Eye.csv')

    # Initialize analyzer
    analyzer = TOSPAnalyzer()

    # Process data
    processed_data = analyzer.preprocess_data(data)

    # Generate pairs
    pairs = analyzer.generate_pairs(processed_data)

    # Train model
    performance_report = analyzer.train_model(pairs)

    # Identify suspicious pairs
    suspicious = analyzer.identify_suspicious_pairs(pairs, threshold=0.7)

    # Print comprehensive analysis
    analyzer.print_analysis_results(processed_data, pairs, suspicious)
