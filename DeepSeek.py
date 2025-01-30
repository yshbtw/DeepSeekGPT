from groq import Groq
import os
from typing import List, Dict
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, Response, stream_with_context

load_dotenv()


api_key = os.getenv('GROQ_API_KEY')
if not api_key:
    raise ValueError("Please set GROQ_API_KEY in your .env file")

app = Flask(__name__)
chatbot = None

class ChatBot:
    def __init__(self):
        self.client = Groq(api_key=api_key)
        self.conversation_history: List[Dict[str, str]] = []
        self.system_message = {
            "role": "system",
            "content": "You are a helpful AI assistant that provides accurate and concise responses."
        }
        self.conversation_history.append(self.system_message)

    def get_response(self, user_input: str):

        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })

        try:

            completion = self.client.chat.completions.create(
                model="deepseek-r1-distill-llama-70b",
                messages=self.conversation_history,
                temperature=0.6,
                max_tokens=4096,
                top_p=0.95,
                stream=True,
                stop=None,
            )

            def generate():
                assistant_response = ""
                
                for chunk in completion:
                    content = chunk.choices[0].delta.content or ""
                    assistant_response += content
                    yield content


                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_response
                })

            return generate()

        except Exception as e:
            return str(e)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    global chatbot
    if chatbot is None:
        chatbot = ChatBot()
    
    user_message = request.json.get('message', '')
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    return Response(
        stream_with_context(chatbot.get_response(user_message)),
        content_type='text/event-stream'
    )

if __name__ == "__main__":
    app.run(debug=True)