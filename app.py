from flask import *
import joblib
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import numpy as np
from os import listdir
from os.path import isfile, join
# from your_module_name import db

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///sentiment_history.db'
db = SQLAlchemy(app)

# create a database model
class SentimentResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    input_text = db.Column(db.String(500))
    positive_prob = db.Column(db.Float)
    neutral_prob = db.Column(db.Float)
    negative_prob = db.Column(db.Float)
    overall_sentiment = db.Column(db.String(10))
    image_filename = db.Column(db.String(100)) 
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Load the model and vectorizer
model = joblib.load("./Sentiment_Model_CountVectorizer.pkl")
vectorizer = joblib.load("./CountVectorizer.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/admin")
def admin():
    return render_template("admin.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    # Get user input from the form
    input_text = request.form.get("text")
    image_filename = request.form.get("image_filename")

    # # Check if image_filename is None
    # if image_filename is None:
    #     # Handle the case where image_filename is not provided
    #     # You can set a default image or return an error message
    #     # For demonstration purposes, let's set a default image
    #     image_filename = "default-image.jpg"

    # Vectorize the input text
    input_vectorized = vectorizer.transform([input_text])

    # Predict probabilities
    predicted_probabilities = model.predict_proba(input_vectorized)[0]

    # Define sentiment labels
    sentiment_labels = ["Positive", "Neutral", "Negative"]

    # Determine the overall sentiment
    overall_sentiment_index = np.argmax(predicted_probabilities)
    overall_sentiment = sentiment_labels[overall_sentiment_index]

    # Save the results to the database
    result = SentimentResult(
        input_text=input_text,
        positive_prob=predicted_probabilities[sentiment_labels.index("Positive")],
        neutral_prob=predicted_probabilities[sentiment_labels.index("Neutral")],
        negative_prob=predicted_probabilities[sentiment_labels.index("Negative")],
        overall_sentiment=overall_sentiment,
        image_filename=image_filename
    )
    db.session.add(result)
    db.session.commit()

    # Redirect to the results page
    return redirect(url_for("results"))


@app.route("/results")
def results():
    # Retrieve sentiment history from the database
    sentiment_history = SentimentResult.query.all()
    return render_template("results.html", sentiment_history=sentiment_history)


if __name__ == "__main__":
    with app.app_context():
        # db.drop_all()
        # print("records has been dropped")
        # Create the database tables
        db.create_all()
    app.run(debug=True)
