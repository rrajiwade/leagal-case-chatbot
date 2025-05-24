from flask import Flask, request, jsonify
from flask_cors import CORS
import openai

app = Flask(__name__)
CORS(app)

# Set your OpenAI API key here
openai.api_key = "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"  # Replace with your actual API key

@app.route('/')
def home():
    return "ChatGPT chatbot server is running. Use /chat with POST."

@app.route('/chat', methods=['POST'])
def chat():
    try:
        print("Chat route hit")
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'Missing message'}), 400

        user_message = data['message']
        print(user_message)

        response = openai.ChatCompletion.create(
            model="gpt-4",  # or "gpt-3.5-turbo"
            messages=[
                {"role": "system", "content": "You are a helpful legal assistant AI trained to answer questions about legal cases."},
                {"role": "user", "content": user_message}
            ],
            temperature=0.5,
            max_tokens=500
        )

        ai_reply = response['choices'][0]['message']['content']
        return jsonify({'reply': ai_reply})

    except Exception as e:
        print(f"OpenAI API error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
