from flask import Flask, render_template
from flask_socketio import SocketIO

def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["SECRET_KEY"] = "changeme"
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

    @app.get("/")
    def index():
        return render_template("index.html")

    return app, socketio
