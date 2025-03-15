import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os


def ensure_dir(directory):
    os.makedirs(directory, exist_ok=True)


class TOSPAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.classifier = RandomForestClassifier(random_state=42)
        self.processed_data = None
        self.tfidf_matrix = None

    def preprocess_data(self, df):
        """
        Preprocess the TOSP data and create features for analysis
        """
        # Clean description text
        df['Description'] = df['Description'].str.upper()

        # Create procedure type features
        df['procedure_type'] = df['Description'].apply(lambda x: x.split(',')[0])

        # Extract location information (e.g., BILATERAL, UNILATERAL)
        df['is_bilateral'] = df['Description'].str.contains('BILATERAL').astype(int)
        df['is_unilateral'] = df['Description'].str.contains('UNILATERAL').astype(int)

        # Create numerical features
        df['table_numeric'] = df['Table'].apply(lambda x:
                                                float(x[:-1]) if x[0].isdigit() else 0)

        # Store processed data for later use
        self.processed_data = df

        # Create TF-IDF matrix for all descriptions
        self.tfidf_matrix = self.vectorizer.fit_transform(df['Description'])

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
                    self.tfidf_matrix[i:i + 1],
                    self.tfidf_matrix[j:j + 1]
                )[0][0]

                # Check if procedures are related
                same_procedure = int(code1['procedure_type'] == code2['procedure_type'])

                # Check for bilateral/unilateral conflict
                bilateral_conflict = int(
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
                    'is_inappropriate': int(bilateral_conflict and same_procedure)
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

        return classification_report(y_test, y_pred, output_dict=True)

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
        proc_dist = df['procedure_type'].value_counts().head(10)  # Show top 10
        for proc, count in proc_dist.items():
            print(f"{proc}: {count}")
        if df['procedure_type'].nunique() > 10:
            print(f"...and {df['procedure_type'].nunique() - 10} more procedure types")
        print("\n" + "=" * 50 + "\n")

        # 3. Table Distribution
        print("3. Table Distribution:")
        table_dist = df['Table'].value_counts()
        for table, count in table_dist.items():
            print(f"Table {table}: {count}")
        print("\n" + "=" * 50 + "\n")

        # 4. Bilateral/Unilateral Procedures
        print("4. Bilateral/Unilateral Procedures:")
        bilateral = df[df['is_bilateral'] == 1]
        unilateral = df[df['is_unilateral'] == 1]
        print(f"Bilateral procedures: {len(bilateral)}")
        print(f"Unilateral procedures: {len(unilateral)}")

        # Print first 5 of each
        print("\nBilateral Procedures (first 5):")
        for _, row in bilateral.head(5).iterrows():
            print(f"- {row['Code']}: {row['Description']}")
        if len(bilateral) > 5:
            print(f"...and {len(bilateral) - 5} more bilateral procedures")

        print("\nUnilateral Procedures (first 5):")
        for _, row in unilateral.head(5).iterrows():
            print(f"- {row['Code']}: {row['Description']}")
        if len(unilateral) > 5:
            print(f"...and {len(unilateral) - 5} more unilateral procedures")
        print("\n" + "=" * 50 + "\n")

        # 5. Potential Conflicts
        if pairs_df is not None:
            print("5. Code Pair Analysis:")
            print(f"Total number of possible pairs: {len(pairs_df)}")
            print(f"Pairs with high similarity (>0.8): {len(pairs_df[pairs_df['desc_similarity'] > 0.8])}")
            print(f"Pairs with same procedure type: {len(pairs_df[pairs_df['same_procedure'] == 1])}")
            print(f"Pairs with bilateral/unilateral conflict: {len(pairs_df[pairs_df['bilateral_conflict'] == 1])}")
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

    def compare_codes(self, code1, code2):
        """
        Compare two procedure codes and return similarity details
        """
        # Check if codes exist in the dataset
        if code1 not in self.processed_data['Code'].values:
            return {"error": f"Code {code1} not found in dataset"}

        if code2 not in self.processed_data['Code'].values:
            return {"error": f"Code {code2} not found in dataset"}

        # Get descriptions
        desc1 = self.processed_data[self.processed_data['Code'] == code1]['Description'].values[0]
        desc2 = self.processed_data[self.processed_data['Code'] == code2]['Description'].values[0]

        # Get indices
        idx1 = self.processed_data[self.processed_data['Code'] == code1].index[0]
        idx2 = self.processed_data[self.processed_data['Code'] == code2].index[0]

        # Calculate description similarity
        desc_sim = cosine_similarity(
            self.tfidf_matrix[idx1:idx1 + 1],
            self.tfidf_matrix[idx2:idx2 + 1]
        )[0][0]

        # Extract features
        proc1 = self.processed_data.iloc[idx1]['procedure_type']
        proc2 = self.processed_data.iloc[idx2]['procedure_type']
        same_procedure = int(proc1 == proc2)
        table_diff = abs(self.processed_data.iloc[idx1]['table_numeric'] -
                         self.processed_data.iloc[idx2]['table_numeric'])

        # Check for conflicts
        is_bilateral1 = self.processed_data.iloc[idx1]['is_bilateral']
        is_unilateral1 = self.processed_data.iloc[idx1]['is_unilateral']
        is_bilateral2 = self.processed_data.iloc[idx2]['is_bilateral']
        is_unilateral2 = self.processed_data.iloc[idx2]['is_unilateral']

        bilateral_conflict = int(
            (is_bilateral1 and is_unilateral2) or
            (is_unilateral1 and is_bilateral2)
        )

        # Create feature vector for prediction
        features = pd.DataFrame({
            'desc_similarity': [desc_sim],
            'same_procedure': [same_procedure],
            'table_difference': [table_diff]
        })

        # Get fraud probability if model is trained
        fraud_probability = None
        if hasattr(self.classifier, 'predict_proba'):
            fraud_probability = self.classifier.predict_proba(features)[0][1]

        # Prepare result
        result = {
            "code1": code1,
            "code2": code2,
            "description1": desc1,
            "description2": desc2,
            "similarity_score": round(desc_sim, 4),
            "similarity_percentage": f"{round(desc_sim * 100, 2)}%",
            "same_procedure": "Yes" if same_procedure else "No",
            "procedure_type1": proc1,
            "procedure_type2": proc2,
            "table_difference": round(table_diff, 2),
        }

        if bilateral_conflict:
            result["conflict_type"] = "Bilateral/Unilateral conflict detected"

        if fraud_probability is not None:
            result["fraud_probability"] = round(fraud_probability, 4)
            result["fraud_percentage"] = f"{round(fraud_probability * 100, 2)}%"

        return result

    def plot_feature_importance(self, charts_dir):
        """
        Plot feature importance from the Random Forest model
        """
        if not hasattr(self.classifier, 'feature_importances_'):
            print("Model hasn't been trained yet. No feature importance to plot.")
            return

        # Get feature importances
        feature_names = ['Description Similarity', 'Same Procedure', 'Table Difference']
        importances = self.classifier.feature_importances_

        # Create a dataframe for plotting
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        # Create the plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title('Feature Importance in Random Forest Model')
        plt.tight_layout()

        # Save the plot
        save_path = os.path.join(charts_dir, 'RandomForest_feature_importance.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance chart saved to {save_path}")
        plt.close()


def interactive_code_comparison(analyzer):
    print("\n==== Procedure Code Comparison Tool (Random Forest) ====")
    print("Enter two procedure codes to compare their similarity.")
    print("Type 'exit' or 'quit' to return to main program.\n")

    # Get list of valid codes for validation
    valid_codes = analyzer.processed_data['Code'].unique()

    while True:
        # Get first code
        code1 = input("Enter first procedure code (or 'exit' to quit): ").strip()
        if code1.lower() in ['exit', 'quit']:
            break

        # Validate first code
        if code1 not in valid_codes:
            print(f"Error: Code '{code1}' not found in dataset.")
            print(f"Available codes include: {', '.join(valid_codes[:5])}... (and {len(valid_codes) - 5} more)")
            continue

        # Get second code
        code2 = input("Enter second procedure code: ").strip()
        if code2.lower() in ['exit', 'quit']:
            break

        # Validate second code
        if code2 not in valid_codes:
            print(f"Error: Code '{code2}' not found in dataset.")
            print(f"Available codes include: {', '.join(valid_codes[:5])}... (and {len(valid_codes) - 5} more)")
            continue

        # Get comparison result
        result = analyzer.compare_codes(code1, code2)

        # Display result in a formatted way
        print("\n==== Comparison Result (Random Forest Analysis) ====")
        print(f"Code 1: {result['code1']}")
        print(f"Description: {result['description1']}")
        print(f"Procedure Type: {result['procedure_type1']}")
        print("\n")
        print(f"Code 2: {result['code2']}")
        print(f"Description: {result['description2']}")
        print(f"Procedure Type: {result['procedure_type2']}")
        print("\n")
        print(f"Description Similarity Score: {result['similarity_score']}")
        print(f"Similarity Percentage: {result['similarity_percentage']}")
        print(f"Same Procedure Type: {result['same_procedure']}")
        print(f"Table Difference: {result['table_difference']}")

        if "conflict_type" in result:
            print(f"\nWarning: {result['conflict_type']}")

        if "fraud_probability" in result:
            print(f"\nFraud Probability: {result['fraud_probability']}")
            print(f"Fraud Percentage: {result['fraud_percentage']}")

        print("\n" + "=" * 30 + "\n")

        # Ask if user wants to continue
        cont = input("Compare another pair? (y/n): ").strip().lower()
        if cont != 'y':
            break

    print("Exiting code comparison tool.")

def main():
    # Set paths
    charts_dir = '../../3_Results/Archives/RandomForestAnalysis'
    ensure_dir(charts_dir)

    data_path = '../../1_DataPreprocessing/DataSets/CleanedDataset/SL_Eye.csv'
    print(f"Loading data from: {data_path}")
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"File not found: {data_path}")
        print("Please check the path and try again.")
        data_path = input("Enter the correct path to your data file: ")
        data = pd.read_csv(data_path)

    # Initialize analyzer
    analyzer = TOSPAnalyzer()

    # Process data
    processed_data = analyzer.preprocess_data(data)
    print(f"Processed {len(processed_data)} records")

    # Generate pairs for training
    print("Generating code pairs for analysis...")
    pairs_df = analyzer.generate_pairs(processed_data)
    print(f"Generated {len(pairs_df)} pairs")

    # Train model
    print("Training Random Forest model...")
    model_report = analyzer.train_model(pairs_df)
    print("Model Performance:")
    print(f"Accuracy: {model_report['accuracy']:.4f}")
    print(f"Precision: {model_report['1']['precision']:.4f}")
    print(f"Recall: {model_report['1']['recall']:.4f}")
    print(f"F1-Score: {model_report['1']['f1-score']:.4f}")

    # Plot feature importance
    analyzer.plot_feature_importance(charts_dir)

    # Identify suspicious pairs
    suspicious_pairs = analyzer.identify_suspicious_pairs(pairs_df, threshold=0.7)
    print(f"Identified {len(suspicious_pairs)} suspicious pairs")

    # Print analysis results
    analyzer.print_analysis_results(processed_data, pairs_df, suspicious_pairs)

    # Start interactive code comparison
    while True:
        print("\n=== Options ===")
        print("1. Compare specific procedure codes")
        print("2. Exit")

        choice = input("Enter your choice (1-2): ").strip()

        if choice == '1':
            interactive_code_comparison(analyzer)
        elif choice == '2':
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")

if __name__ == '__main__':
    main()