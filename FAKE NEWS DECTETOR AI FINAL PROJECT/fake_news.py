


import pandas as pd
import re, string
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


data = {
    'text': [
        "SHOCKING: Government hiding alien contact for decades, insider reveals truth",
        "Miracle cure discovered: lemon juice cures cancer overnight, they dont want you to know",
        "Exposed: Celebrities control global banking with secret society conspiracy",
        "Vaccine causes autism, leaked documents suppressed by mainstream media",
        "Deep state plotting against citizens, whistleblower exposes cover up",
        "Scientists paid to lie about climate change, shocking emails leaked",
        "You wont believe what big pharma is hiding about cancer cure breakthrough",
        "Illuminati controls mainstream media, real news banned and censored",
        "Federal Reserve raises interest rates by 0.25 percent amid inflation",
        "Researchers publish climate study findings in peer-reviewed Nature journal",
        "Congress passes infrastructure bill allocating funds for highway repairs",
        "University study shows exercise reduces heart disease risk by 30 percent",
        "WHO releases updated vaccination guidelines based on clinical trial data",
        "Scientists discover new deep sea species near Pacific Ocean floor",
        "Health department confirms decline in flu cases after vaccination campaign",
    ],
    'label': [0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1]  
}
df = pd.DataFrame(data)

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)             
    tokens = text.split()
    tokens = [stemmer.stem(t) for t in tokens          
              if t not in stop_words and len(t) > 2]  
    return ' '.join(tokens)

df['clean'] = df['text'].apply(preprocess)
print("Sample preprocessing:")
print(f"  Before: {df['text'][0]}")
print(f"  After : {df['clean'][0]}\n")

X = df['clean']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=500)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

models = {
    'Naive Bayes  (W14)': MultinomialNB(),
    'Decision Tree(W12)': DecisionTreeClassifier(max_depth=5, random_state=42),
    'KNN          (W13)': KNeighborsClassifier(n_neighbors=3, metric='cosine'),
}

print("="*45)
print("MODEL RESULTS + 5-FOLD CROSS VALIDATION (W16)")
print("="*45)
for name, model in models.items():
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)
    acc   = accuracy_score(y_test, preds)
    cv    = cross_val_score(model, X_train_vec, y_train, cv=3).mean()
    print(f"\n{name}")
    print(f"  Test Accuracy : {acc:.2f}")
    print(f"  CV  Accuracy  : {cv:.2f}")
    print(classification_report(y_test, preds, target_names=['Fake','Real'], zero_division=0))

print("="*45)
print("CUSTOM PREDICTIONS")
print("="*45)
best_model = models['Naive Bayes  (W14)']

headlines = [
    "Scientists confirm vaccine is safe after large clinical trial",
    "SHOCKING truth exposed: Government hiding alien contact",
]
for h in headlines:
    vec  = vectorizer.transform([preprocess(h)])
    pred = best_model.predict(vec)[0]
    print(f"  {'REAL ✅' if pred == 1 else 'FAKE 🔴'}  —  {h}")
