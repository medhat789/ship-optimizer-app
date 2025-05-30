try:
    from flask import Flask
    print("Flask imported successfully!")
    app = Flask(__name__)
    print("Flask app initialized!")
    print("Test script completed successfully.")
except ImportError as e:
    print(f"Error importing Flask: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
