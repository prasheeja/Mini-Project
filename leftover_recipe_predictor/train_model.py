# train_model.py
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import pickle, os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "recipe.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")  # now binary encoder
# ENCODER_PATH not needed anymore

# Load dataset
df = pd.read_csv(DATA_PATH)

# Ensure ingredients column exists
if "Ingredients" not in df.columns or "Recipe Name" not in df.columns:
    raise Exception("recipe.csv must have 'Ingredients' and 'Recipe Name' columns")

# Handle missing values by replacing them with an empty string
df['Ingredients'].fillna('', inplace=True)

# Preprocess ingredients: split by comma and strip spaces
df["Ingredients_list"] = df["Ingredients"].apply(lambda x: [i.strip().lower() for i in x.split(",")])

# Use MultiLabelBinarizer to convert ingredients to binary features
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df["Ingredients_list"])
y = df["Recipe Name"]

# Train Decision Tree
model = DecisionTreeClassifier()
model.fit(X, y)

# Save model and vectorizer (binary encoder)
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

with open(VECTORIZER_PATH, "wb") as f:
    pickle.dump(mlb, f)

print("✅ Model and binary encoder saved!")