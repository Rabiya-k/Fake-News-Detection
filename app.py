import numpy as np
from flask import Flask, request, render_template
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)        # Initialize the Flask application
vectorizer = TfidfVectorizer()  # Initialize a TfidfVectorizer

model = pickle.load(open('model.pkl', 'rb'))      # Load the trained model
vect = pickle.load(open('tfidfvect2.pkl', 'rb'))  # Load the TF-IDF vectorizer

@app.route('/')   # Define a route for the homepage
def home():
    return render_template('index.html', News_prediction='')

@app.route('/predict', methods=['POST'])   # Define a route for prediction
def predict():
    if request.method == 'POST':
        news = request.form['news']    # Extract the news from the form data
        input_data = [news]
        vectorized_input_data = vect.transform(input_data)  # Vectorize the input news using the loaded TF-IDF vectorizer
        prediction = model.predict(vectorized_input_data)   # Make prediction using the loaded model
        if prediction[0] == 0:
            result = "Looks Like a FAKE News"
        else:
            result = "Looks Like a REAL News"
        return render_template('index.html', News_prediction=result, news=news) # Render the result template with the prediction and input news
    else:
        return render_template('index.html', News_prediction='')   # If request method is not POST, render the homepage

if __name__ == "__main__":
    app.run(debug=True)    # Run the Flask application