from flask import Flask, render_template, request
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model and vectorizer
vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("spam_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get email text from form
    email_text = request.form['email']

    # Transform input text using the SAME trained vectorizer
    input_data = vectorizer.transform([email_text])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Interpret result (1 = Spam, 0 = Not Spam)
    result = "🚫 Spam" if prediction == 0 else "✅ Not Spam"

    # Return result to HTML page
    return render_template('index.html', prediction=result, email=email_text)

if __name__ == '__main__':
    app.run(debug=True)
