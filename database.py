from flask_sqlalchemy import SQLAlchemy
from flask import current_app

db = SQLAlchemy()

def start_db(app):
    db.init_app(app)
    with app.app_context(): # This creates an applicatoin context for db interactions with the current session
        db.create_all() # checks the database schema and adds any tables that are missing
    return db