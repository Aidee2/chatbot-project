from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import nltk, random, string, re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

app = Flask(__name__, template_folder='template')
CORS(app)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

training_data = [
    ("hello", "greeting"),
    ("hi", "greeting"),
    ("how are you", "greeting"),
    ("bye", "farewell"),
    ("goodbye", "farewell"),
    ("thanks", "thanks"),
    ("thank you", "thanks"),
    ("tell me about your product", "product_info"),
    ("what are your hours", "hours"),
    ("when are you open", "hours")
]

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and t not in string.punctuation]
    return " ".join(tokens)

texts = [preprocess(sentence) for sentence, intent in training_data]
labels = [intent for _, intent in training_data]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
model = LogisticRegression()
model.fit(X, labels)

responses = {
    "greeting": ["Hello! How can I help you?", "Hi there! What can I do for you?"],
    "farewell": ["Goodbye! Have a great day!", "See you soon!"],
    "thanks": ["You're welcome!", "Glad to help!"],
    "product_info": ["We offer AI-based products and learning tools."],
    "hours": ["We are open from 9 AM to 6 PM, Monday to Friday."]
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    user_message = request.form["message"]
    processed = preprocess(user_message)
    X_test = vectorizer.transform([processed])
    intent = model.predict(X_test)[0]
    reply = random.choice(responses[intent])
    return jsonify({"response": reply})

if __name__ == "__main__":
    app.run(debug=True)
