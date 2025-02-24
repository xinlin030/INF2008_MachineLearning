import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


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
        df['word_count'] = df['Description'].str.split().str.len()

        # Convert table to numeric
        df['table_numeric'] = df['Table'].apply(lambda x:
                                                float(x[:-1]) if x[0].isdigit() else 0)

        return df

    def create_pair_features(self, df):
        """
        Create features for procedure pairs
        """
        pairs = []
        descriptions = df['Description'].values

        # Get TF-IDF features
        tfidf_matrix = self.vectorizer.fit_transform(descriptions)

        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                # Calculate TF-IDF similarity
                similarity = np.dot(tfidf_matrix[i].toarray(),
                                    tfidf_matrix[j].toarray().T)[0][0]

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

    def plot_feature_importance(self, feature_importance):
        """
        Plot feature importance
        """
        plt.figure(figsize=(10, 6))
        plt.bar(feature_importance['feature'], feature_importance['importance'])
        plt.title('Feature Importance in XGBoost Model')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

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


# Example usage
if __name__ == "__main__":
    # Load data
    data = pd.read_csv('DataSets/CleanedDataset/combined_dataset.csv')

    # Initialize model
    xgb_model = TOSPXGBoost()

    # Preprocess data
    processed_data = xgb_model.preprocess_data(data)

    # Create pair features
    pairs = xgb_model.create_pair_features(processed_data)

    # Train model
    metrics, feature_importance = xgb_model.train_model(pairs)

    # Print results
    print("\n=== Model Performance ===")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")

    print("\n=== Feature Importance ===")
    print(feature_importance)

    # Plot feature importance
    xgb_model.plot_feature_importance(feature_importance)

    # Get predictions
    suspicious_pairs = xgb_model.predict_inappropriate_pairs(pairs)

    print("\n=== Top Suspicious Pairs ===")
    for _, row in suspicious_pairs.head().iterrows():
        print(f"\nPair: {row['code1']} - {row['code2']}")
        print(f"Fraud Probability: {row['fraud_probability']:.2f}")
        print(f"Same Procedure: {'Yes' if row['same_procedure'] else 'No'}")
        print(f"Bilateral Conflict: {'Yes' if row['bilateral_conflict'] else 'No'}")