# initialize the database with models/run this once before running the Flask app
from app2 import app, db
from models import User, Document, TranscriptionSession

def init_db():
    with app.app_context():
        db.create_all()
        print("Database tables created successfully.")
    

if __name__ == '__main__':
    init_db()
