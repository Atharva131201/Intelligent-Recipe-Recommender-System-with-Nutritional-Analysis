import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from IPython.display import display, Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sqlalchemy import create_engine

app = Flask(__name__)

# Assuming 'conn' is the connection to your SQLite database
engine = create_engine('sqlite:///recipes.db')
conn = engine.connect()

# Assuming 'conn' is the connection to your SQLite database
query = "SELECT * FROM recipes_table;"
df = pd.read_sql(query, conn)

# Select relevant features for content-based filtering
features = ['ingredients', 'dishTypes', 'calories']

# Combine selected features into a single text column
df['combined_features'] = df[features].astype(str).apply(lambda x: ' '.join(x), axis=1)

# Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Define the path to the feedback CSV file
feedback_csv_path = 'feedback.csv'

# Function to get recipe recommendations based on content similarity
def get_content_recommendations(title, user_ingredients):
    idx = df[df['title'] == title].index[0]
    user_ingredients_lower = [ingredient.lower() for ingredient in user_ingredients]

    # Filter recipes that include at least one of the user ingredients
    candidate_indices = df[df['ingredients'].apply(lambda x: any(ingredient in x.lower() for ingredient in user_ingredients_lower))].index

    # Calculate similarity scores only for the candidate recipes
    sim_scores = [(i, cosine_sim[idx, i]) for i in candidate_indices]
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the top 5 similar recipes
    sim_scores = sim_scores[1:6] if len(sim_scores) > 6 else sim_scores[1:]
    
    recipe_indices = [x[0] for x in sim_scores]
    return df['title'].iloc[recipe_indices]

# Function to recommend recipes based on user input
def recommend_recipes():
    # Take user inputs
    user_ingredients = request.form.get('ingredients').split(', ')
    user_max_price = float(request.form.get('max_price'))
    user_dish_type = request.form.get('dish_type')
    user_is_vegetarian = request.form.get('is_vegetarian').lower() == 'yes'
    user_max_minutes = int(request.form.get('max_minutes'))

    # Filter recipes based on user input
    filtered_df = df[
        (df['pricePerServing'] <= user_max_price) &
        (df['dishTypes'].str.contains(user_dish_type, case=False)) &
        (df['vegetarian'] == user_is_vegetarian) &
        (df['readyInMinutes'] <= user_max_minutes) &
        (df['ingredients'].apply(lambda x: all(ingredient.lower() in x.lower() for ingredient in user_ingredients)))
    ]

    # Sort the filtered recipes by healthScore
    sorted_df = filtered_df.sort_values(by='healthScore', ascending=False)

    # Get the top content-based recommendation
    if not sorted_df.empty:
        top_recipe = sorted_df.iloc[0]
        content_recommendations = get_content_recommendations(top_recipe['title'], user_ingredients)
        return top_recipe, content_recommendations
    else:
        return None, None

# Function to display image and recipe details
def display_recipe(recipe):
    display(Image(url=recipe['image']))
    print("Title:", recipe['title'])
    print("Calories:", recipe['calories'])
    print("Health Score:", recipe['healthScore'])
    print("Percent Protein:", recipe['percentProtein'])
    print("Fat/g:", recipe['Fat/g'])
    print("Sugar/g:", recipe['Sugar/g'])
    print("Cholesterol/mg:", recipe['Cholesterol/mg'])
    # Add other columns as needed

# Function to store feedback in the CSV file
def store_feedback_csv(title, ingredients, price_per_serving, vegetarian, ready_in_minutes):
    feedback_data = {
        'title': title,
        'ingredients': ingredients,
        'pricePerServing': price_per_serving,
        'vegetarian': vegetarian,
        'readyInMinutes': ready_in_minutes
    }
    feedback_df = pd.DataFrame([feedback_data])

    # If the CSV file doesn't exist, create it with the header
    if not os.path.isfile(feedback_csv_path):
        feedback_df.to_csv(feedback_csv_path, index=False, header=True)
    else:
        # Append the feedback data to the CSV file
        feedback_df.to_csv(feedback_csv_path, mode='a', index=False, header=False)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        top_recipe, content_recommendations = recommend_recipes()
        if top_recipe is not None:
            return render_template('index.html', top_recipe=top_recipe, content_recommendations=content_recommendations, df=df)
        else:
            return render_template('feedback.html')
    return render_template('index.html', top_recipe=None, content_recommendations=None, df=df)

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        title = request.form.get('title')
        ingredients = request.form.get('ingredients')
        cost = request.form.get('cost')
        veg_non_veg = request.form.get('veg_non_veg')
        time = request.form.get('time')

        # Save feedback to the feedback.csv file
        store_feedback_csv(title, ingredients, cost, veg_non_veg.lower() == 'vegetarian', time)

        return redirect(url_for('home'))
    
    return render_template('feedback.html')

if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')
