import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_model():
    # Load and prepare the dataset
    data = pd.read_csv(r"C:\Users\eshin\Downloads\ISProject\pythonProject\labeled_data.csv")
    data = data[['tweet', 'class']]
    data.columns = ['text', 'label']
    data['label'] = data['label'].apply(lambda x: 1 if x == 1 else 0)  # 1 for offensive, 0 for non-offensive

    # Split data into training and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        data['text'],
        data['label'],
        test_size=0.2,
        random_state=42
    )

    # Vectorize text data with n-grams (bigrams and trigrams)
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)

    # Train logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, train_labels)

    # Evaluate the model
    val_predictions = model.predict(X_val)
    accuracy = accuracy_score(val_labels, val_predictions)
    print(f"Validation Accuracy: {accuracy:.2f}")
    print(classification_report(val_labels, val_predictions))

    # Save model and vectorizer
    joblib.dump(model, 'logistic_regression_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

def load_model():
    try:
        model = joblib.load('logistic_regression_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        print("Model or Vectorizer not found!")
        return None, None

def predict_message(message, model, vectorizer, threshold=0.635):
    """
    Predict whether the message is offensive or non-offensive, adjusting the decision threshold.
    """
    # Transform the message to the feature vector using the saved vectorizer
    message_vector = vectorizer.transform([message])

    # Get predicted probabilities for the positive class (offensive)
    message_probability = model.predict_proba(message_vector)[:, 1]  # Probability for offensive class

    # Adjust decision threshold
    if message_probability[0] >= threshold:
        print("Message classified as offensive.")
        return True  # Offensive message
    else:
        print("Message classified as non-offensive.")
        return False  # Non-offensive message


if __name__ == '__main__':
    # Uncomment the line below to train the model
    train_model()
    pass
