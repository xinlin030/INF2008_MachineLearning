import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os


def ensure_dir(directory):
    os.makedirs(directory, exist_ok=True)


class TOSPXGBoost:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=100)
        self.scaler = StandardScaler()
        self.model = xgb.XGBClassifier(
            learning_rate=0.1,
            max_depth=4,
            n_estimators=100,
            random_state=42
        )
        self.processed_data = None
        self.tfidf_matrix = None

    def preprocess_data(self, df):
        """
        Preprocess the TOSP data and create features
        """
        # Clean description text
        df['Description'] = df['Description'].str.upper()

        # Extract features
        df['procedure_type'] = df['Description'].apply(lambda x: x.split(',')[0])
        df['is_bilateral'] = df['Description'].str.contains('BILATERAL').astype(int)
        df['is_unilateral'] = df['Description'].str.contains('UNILATERAL').astype(int)
        df['word_count'] = df['Description'].apply(lambda x: len(x.split()))

        # Convert table to numeric
        df['table_numeric'] = df['Table'].apply(lambda x:
                                                float(x[:-1]) if x[0].isdigit() else 0)

        self.processed_data = df
        return df

    def create_pair_features(self, df):
        """
        Create features for procedure pairs
        """
        pairs = []
        descriptions = df['Description'].values

        # Get TF-IDF features
        self.tfidf_matrix = self.vectorizer.fit_transform(descriptions)

        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                # Calculate TF-IDF similarity
                similarity = np.dot(self.tfidf_matrix[i].toarray(),
                                    self.tfidf_matrix[j].toarray().T)[0][0]

                # Extract features
                same_procedure = int(df.iloc[i]['procedure_type'] ==
                                     df.iloc[j]['procedure_type'])
                table_diff = abs(df.iloc[i]['table_numeric'] -
                                 df.iloc[j]['table_numeric'])
                word_count_diff = abs(df.iloc[i]['word_count'] -
                                      df.iloc[j]['word_count'])
                bilateral_conflict = int(
                    (df.iloc[i]['is_bilateral'] and df.iloc[j]['is_unilateral']) or
                    (df.iloc[i]['is_unilateral'] and df.iloc[j]['is_bilateral'])
                )

                # Create target (you would normally have this from historical data)
                is_inappropriate = int(bilateral_conflict and same_procedure)

                pairs.append({
                    'code1': df.iloc[i]['Code'],
                    'code2': df.iloc[j]['Code'],
                    'similarity': similarity,
                    'same_procedure': same_procedure,
                    'table_diff': table_diff,
                    'word_count_diff': word_count_diff,
                    'bilateral_conflict': bilateral_conflict,
                    'is_inappropriate': is_inappropriate
                })

        return pd.DataFrame(pairs)

    def train_model(self, pairs_df):
        """
        Train XGBoost model and evaluate performance
        """
        # Prepare features
        features = ['similarity', 'same_procedure', 'table_diff',
                    'word_count_diff', 'bilateral_conflict']
        X = pairs_df[features]
        y = pairs_df['is_inappropriate']

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        # Train model
        self.model.fit(X_train, y_train)

        # Make predictions
        y_pred = self.model.predict(X_test)

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }

        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return metrics, feature_importance

    def plot_feature_importance(self, feature_importance, charts_dir):
        """
        Plot feature importance and save to file
        """
        save_path = os.path.join(charts_dir, 'XGBoost_feature_importance.png')

        plt.figure(figsize=(10, 6))
        plt.bar(feature_importance['feature'], feature_importance['importance'])
        plt.title('Feature Importance in XGBoost Model')
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save chart to file
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance chart saved to {save_path}")
        plt.close()  # Close the figure to free memory

    def predict_inappropriate_pairs(self, pairs_df, threshold=0.7):
        """
        Predict inappropriate pairs using trained model
        """
        features = ['similarity', 'same_procedure', 'table_diff',
                    'word_count_diff', 'bilateral_conflict']
        X = pairs_df[features]
        X_scaled = self.scaler.transform(X)

        # Get probability predictions
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        pairs_df['fraud_probability'] = probabilities

        # Filter suspicious pairs
        suspicious_pairs = pairs_df[pairs_df['fraud_probability'] > threshold]

        return suspicious_pairs.sort_values('fraud_probability', ascending=False)

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

        # Calculate TF-IDF similarity
        similarity = np.dot(self.tfidf_matrix[idx1].toarray(),
                            self.tfidf_matrix[idx2].toarray().T)[0][0]

        # Extract features
        proc1 = self.processed_data.iloc[idx1]['procedure_type']
        proc2 = self.processed_data.iloc[idx2]['procedure_type']
        same_procedure = int(proc1 == proc2)
        table_diff = abs(self.processed_data.iloc[idx1]['table_numeric'] -
                         self.processed_data.iloc[idx2]['table_numeric'])
        word_count_diff = abs(self.processed_data.iloc[idx1]['word_count'] -
                              self.processed_data.iloc[idx2]['word_count'])

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
        features = np.array([[
            similarity, same_procedure, table_diff,
            word_count_diff, bilateral_conflict
        ]])

        # Scale features
        features_scaled = self.scaler.transform(features)

        # Get fraud probability if model is trained
        fraud_probability = None
        if hasattr(self.model, 'predict_proba'):
            fraud_probability = self.model.predict_proba(features_scaled)[0][1]

        # Prepare result
        result = {
            "code1": code1,
            "code2": code2,
            "description1": desc1,
            "description2": desc2,
            "similarity_score": round(similarity, 4),
            "similarity_percentage": f"{round(similarity * 100, 2)}%",
            "same_procedure": "Yes" if same_procedure else "No",
            "table_difference": round(table_diff, 2),
            "word_count_difference": word_count_diff,
        }

        if bilateral_conflict:
            result["conflict_type"] = "Bilateral/Unilateral conflict detected"

        if fraud_probability is not None:
            result["fraud_probability"] = round(fraud_probability, 4)
            result["fraud_percentage"] = f"{round(fraud_probability * 100, 2)}%"

        return result


def interactive_code_comparison(xgb_model):
    print("\n==== Procedure Code Comparison Tool (XGBoost) ====")
    print("Enter two procedure codes to compare their similarity.")
    print("Type 'exit' or 'quit' to return to main program.\n")

    # Get list of valid codes for validation
    valid_codes = xgb_model.processed_data['Code'].unique()

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
        result = xgb_model.compare_codes(code1, code2)

        # Display result in a formatted way
        print("\n==== Comparison Result (XGBoost Analysis) ====")
        print(f"Code 1: {result['code1']}")
        print(f"Description: {result['description1']}")
        print("\n")
        print(f"Code 2: {result['code2']}")
        print(f"Description: {result['description2']}")
        print("\n")
        print(f"TF-IDF Similarity Score: {result['similarity_score']}")
        print(f"Similarity Percentage: {result['similarity_percentage']}")
        print(f"Same Procedure Type: {result['same_procedure']}")
        print(f"Table Difference: {result['table_difference']}")
        print(f"Word Count Difference: {result['word_count_difference']}")

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

    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get script's directory
    charts_dir = script_dir  # Save outputs in the same directory
    ensure_dir(charts_dir)

    data_path = '../../../1_DataPreprocessing/DataSets/CleanedDataset/SL_Eye.csv'
    print(f"Loading data from: {data_path}")
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"File not found: {data_path}")
        print("Please check the path and try again.")
        data_path = input("Enter the correct path to your data file: ")
        data = pd.read_csv(data_path)

    # Initialize model
    xgb_model = TOSPXGBoost()

    # Preprocess data
    processed_data = xgb_model.preprocess_data(data)
    print(f"Processed {len(processed_data)} records")

    # Create pair features
    pairs = xgb_model.create_pair_features(processed_data)
    print(f"Created {len(pairs)} procedure pairs for analysis")

    # Train model
    print("\nTraining XGBoost model...")
    metrics, feature_importance = xgb_model.train_model(pairs)

    # Print results
    print("\n=== Model Performance ===")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")

    print("\n=== Feature Importance ===")
    print(feature_importance)

    # Plot feature importance
    xgb_model.plot_feature_importance(feature_importance, charts_dir)

    # Get predictions
    threshold = 0.7
    suspicious_pairs = xgb_model.predict_inappropriate_pairs(pairs, threshold)

    print(f"\n=== Top Suspicious Pairs (Threshold: {threshold}) ===")
    if len(suspicious_pairs) > 0:
        for _, row in suspicious_pairs.head().iterrows():
            desc1 = processed_data.loc[processed_data['Code'] == row['code1'], 'Description'].values[0]
            desc2 = processed_data.loc[processed_data['Code'] == row['code2'], 'Description'].values[0]

            print(f"\nPair: {row['code1']} ({desc1}) and {row['code2']} ({desc2})")
            print(f"Fraud Probability: {row['fraud_probability']:.2f}")
            print(f"Same Procedure: {'Yes' if row['same_procedure'] else 'No'}")
            print(f"Bilateral Conflict: {'Yes' if row['bilateral_conflict'] else 'No'}")
    else:
        print("No suspicious pairs found above the threshold.")

    # Start interactive code comparison
    while True:
        print("\n=== Options ===")
        print("1. Compare specific procedure codes")
        print("2. Exit")

        choice = input("Enter your choice (1-2): ").strip()

        if choice == '1':
            interactive_code_comparison(xgb_model)
        elif choice == '2':
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")

if __name__ == '__main__':
    main()