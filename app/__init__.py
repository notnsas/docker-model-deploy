from flask import Flask, flash, request, redirect, url_for
from config import Config
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import os

UPLOAD_FOLDER = '/path/to/the/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

app = Flask(__name__)

# from app import routes
app.config.from_object(Config)
db = SQLAlchemy(app)
migrate = Migrate(app, db)


from app import routes, models
with app.app_context():
    db.create_all()
    