from flask import Flask, flash, request, redirect, url_for
from config import Config
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_cors import CORS  # <--- Import this
import os

# Get the directory where this script is running
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Create the upload folder INSIDE your project structure
# This will result in: /home/nara/Coding/Python/DockerModelDeploy/app/uploads
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')

# Now this line will work because you own the folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

app = Flask(__name__)
CORS(app)  # <--- Add this line to enable CORS for all routes

# from app import routes
app.config.from_object(Config)
db = SQLAlchemy(app)
migrate = Migrate(app, db)


from app import routes, models
with app.app_context():
    db.create_all()
    
