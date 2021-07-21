from tika.tika import main
from src.train import predict
import sys

if __name__ == "__main__":
    file_path = sys.argv[1]
    print(predict(file_path))
