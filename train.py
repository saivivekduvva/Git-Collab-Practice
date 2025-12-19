from preprocess import preprocess_data
from model import build_model

def train():
    X, y = preprocess_data()
    model = build_model(X, y)

if __name__ == "__main__":
    train()
