# app.py
from flask import Flask, render_template, request
import pandas as pd
import pickle, os

app = Flask(__name__)

# ----------------- Paths -----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "recipe.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

# ----------------- Load model and vectorizer -----------------
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(VECTORIZER_PATH, "rb") as f:
    mlb = pickle.load(f)

# ----------------- Load dataset -----------------
recipes_df = pd.read_csv(DATA_PATH)

# ----------------- Routes -----------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    ingredients_input = request.form['ingredients']
    input_list = [i.strip().lower() for i in ingredients_input.split(",")]
    input_vector = mlb.transform([input_list])

    # Optional: top 3 recipes by matching ingredients
    def match_score(row):
        if pd.isna(row):
            return 0
        return len(set(input_list) & set([i.strip().lower() for i in str(row).split(",")]))

    recipes_df["score"] = recipes_df["Ingredients"].apply(match_score)
    top_recipes_df = recipes_df.sort_values(by="score", ascending=False).head(3)
    top_recipes = top_recipes_df.to_dict(orient="records")

    return render_template('result.html', recipes=top_recipes, ingredients=ingredients_input)


@app.route('/recipe/<name>')
def recipe_detail(name):
    recipe = recipes_df[recipes_df["Recipe Name"] == name].iloc[0].to_dict()
    return render_template('detail.html', recipe=recipe)


# ----------------- Feedback feature (POST) -----------------
@app.route('/feedback', methods=['POST'])
def feedback_post():
    # You can still get the recipe name if needed
    recipe_name = request.form.get('recipe_name', '')

    # Simple fixed message for all feedback
    feedback_text = "Thank you for your feedback!"

    return render_template('thank_you.html', feedback=feedback_text)



# ----------------- Run App -----------------
if __name__ == '__main__':
    app.run(debug=True)
