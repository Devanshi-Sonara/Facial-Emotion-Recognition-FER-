# main.py

from utils.data_loader import get_data_loaders
from utils.data_augmentation import get_augmenter
from utils.trainer import train_model
from utils.evaluator import evaluate_model
from models.ann import build_ann
from models.cnn import build_cnn
from config import *

# Get data loaders
train_loader, test_loader = get_data_loaders()

#  Build and train ANN
ann_model = build_ann()
train_model(ann_model, train_loader, test_loader, optimizer="adam")
evaluate_model(ann_model, test_loader, model_name="ANN", optimizer_name="Adam", epochs=50)

# ann_model = build_ann()
# train_model(ann_model, train_loader, test_loader, optimizer="sgd")
# evaluate_model(ann_model, test_loader, model_name="ANN", optimizer_name="SGD", epochs=20)

# # Build and train CNN
# cnn_model = build_cnn()
# train_model(cnn_model, train_loader, test_loader, optimizer="adam")
# evaluate_model(cnn_model, test_loader, model_name="CNN", optimizer_name="ADAM", epochs=20)
# cnn_model = build_cnn()
# train_model(cnn_model, train_loader, test_loader, optimizer="sgd")
# evaluate_model(cnn_model, test_loader, model_name="CNN", optimizer_name="SDG", epochs=20)