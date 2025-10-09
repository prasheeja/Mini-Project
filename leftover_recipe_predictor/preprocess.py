import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import pickle

# Load dataset
data = pd.read_csv("data/recipe.csv")

# Combine recipe name + ingredients for training features
data["text"] =  data["Ingredients"]

# Create Bag of Words model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data["text"])

# Save vectorizer for later use in Flask app
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Preprocessing complete. 'vectorizer.pkl' saved.")
