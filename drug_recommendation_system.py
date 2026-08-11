import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import learning_curve
import numpy as np

# ---------------------------------
# Download NLTK Resources
# ---------------------------------
nltk.download('stopwords')
nltk.download('wordnet')

# ---------------------------------
# 1. Load Training and Testing Datasets
# ---------------------------------
train_df = pd.read_csv(r"C:\drugproject\Drug Reviews (Druglib.com)\drugLibTrain_raw.csv")
test_df = pd.read_csv(r"C:\drugproject\Drug Reviews (Druglib.com)\drugLibTest_raw.csv")

print("Training Dataset Loaded Successfully")
print(train_df.head())
print("\nTraining Dataset Shape:", train_df.shape)

print("\nTesting Dataset Loaded Successfully")
print(test_df.head())
print("\nTesting Dataset Shape:", test_df.shape)

# ---------------------------------
# 2. Sentiment Labeling
# ---------------------------------
def get_sentiment(rating):
    if rating >= 8:
        return "positive"
    elif rating >= 5:
        return "neutral"
    else:
        return "negative"

train_df['sentiment'] = train_df['rating'].apply(get_sentiment)
test_df['sentiment'] = test_df['rating'].apply(get_sentiment)

print("\nTraining Sentiment Distribution:")
print(train_df['sentiment'].value_counts())

print("\nTesting Sentiment Distribution:")
print(test_df['sentiment'].value_counts())

# ---------------------------------
# 3. Text Preprocessing
# ---------------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', ' ', str(text))
    text = text.lower()

    words = []
    for word in text.split():
        if word not in stop_words and len(word) > 2:
            words.append(lemmatizer.lemmatize(word))

    return " ".join(words)

train_df['cleaned_review'] = train_df['commentsReview'].apply(clean_text)
test_df['cleaned_review'] = test_df['commentsReview'].apply(clean_text)

# Remove empty reviews
train_df = train_df[train_df['cleaned_review'].str.strip() != ""]
test_df = test_df[test_df['cleaned_review'].str.strip() != ""]
# ---------------------------------
# Text Preprocessing Completed
# ---------------------------------

print("\nText Preprocessing Completed Successfully")

print("\nSample Cleaned Reviews:")
print(train_df[['commentsReview', 'cleaned_review']].head())

# ---------------------------------
# 4. TF-IDF Feature Extraction
# ---------------------------------
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,2),
    min_df=5,
    max_df=0.85,
    sublinear_tf=True
)

X_train = vectorizer.fit_transform(train_df['cleaned_review'])
X_test = vectorizer.transform(test_df['cleaned_review'])

y_train = train_df['sentiment']
y_test = test_df['sentiment']

print("\nTF-IDF Feature Matrix")
print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# ---------------------------------
# 5. Train Multinomial Naïve Bayes
# ---------------------------------
model = MultinomialNB(alpha=0.1)
model.fit(X_train, y_train)
# ---------------------------------
# Accuracy Graph (Learning Curve)
# ---------------------------------

train_sizes, train_scores, test_scores = learning_curve(
    model,
    X_train,
    y_train,
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 5),
    scoring='accuracy'
)

train_accuracy = train_scores.mean(axis=1)
validation_accuracy = test_scores.mean(axis=1)

plt.figure(figsize=(7,5))

plt.plot(train_sizes, train_accuracy,
         marker='o',
         linewidth=2,
         label='Training Accuracy')

plt.plot(train_sizes, validation_accuracy,
         marker='s',
         linewidth=2,
         label='Validation Accuracy')

plt.title("Model Accuracy Graph")
plt.xlabel("Training Samples")
plt.ylabel("Accuracy")
plt.ylim(0, 1.05)
plt.grid(True)
plt.legend()

plt.show()

print("\nModel Trained Successfully")

# ---------------------------------
# 6. Model Evaluation
# ---------------------------------
y_pred = model.predict(X_test)

print("\nPredicted Sentiment Distribution:")
print(pd.Series(y_pred).value_counts())

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(round(accuracy*100,2),"%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Negative', 'Neutral', 'Positive'],
            yticklabels=['Negative', 'Neutral', 'Positive'])

plt.title("Confusion Matrix")
plt.xlabel("Predicted Sentiment")
plt.ylabel("Actual Sentiment")
plt.tight_layout()
plt.show()

# ---------------------------------
# 7. Drug Recommendation
# ---------------------------------
def recommend_drugs(condition):

    condition_df = train_df[
    train_df['condition'].str.lower().str.strip() ==
    condition.lower().strip()
]

    if condition_df.empty:
        print("\nNo records found for this condition.")
        return

    sentiment_counts = condition_df['sentiment'].value_counts()

    plt.figure(figsize=(6,6))
    plt.pie(
        sentiment_counts,
        labels=sentiment_counts.index,
        autopct='%1.1f%%'
    )
    plt.title(f"Sentiment Distribution for {condition}")
    plt.show()

    top_drugs = (
        condition_df[condition_df['sentiment']=="positive"]
        .groupby('urlDrugName')
        .size()
        .sort_values(ascending=False)
        .head(5)
    )

    if top_drugs.empty:
        print("\nNo positive drug recommendations found.")
        return

    print(f"\nTop Recommended Drugs for {condition}\n")

    max_count = top_drugs.max()

    for i,(drug,count) in enumerate(top_drugs.items(),1):

        score = count/max_count

        stars = "★"*round(score*5)+"☆"*(5-round(score*5))

        print(f"{i}. {drug}")
        print(f"   Score : {score:.2f}")
        print(f"   Rating: {stars}\n")

# ---------------------------------
# 8. User Input
# ---------------------------------
while True:

    condition = input("\nEnter Disease/Condition (type 'exit' to quit): ").strip()

    if condition.lower() == "exit":
        print("Thank you!")
        break

    condition_df = train_df[
        train_df['condition'].str.lower().str.strip() ==
        condition.lower().strip()
    ]

    if condition_df.empty:
        print("\n❌ No records found. Please enter a valid disease/condition.")
    else:
        recommend_drugs(condition)


