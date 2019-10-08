from flask import Flask, render_template, request
import requests

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    response = requests.post("http://127.0.0.1:5001/assistant", data=userText)
    print(response.text)
    return str(response.text)

if __name__ == "__main__":
    app.run()
