import argparse
import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms
from datasets import load_from_disk

# -------------------------------------------------
# Arguments
# -------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default="imagenet_model_3.onnx")
    return parser.parse_args()


# -------------------------------------------------
# Load class names
# -------------------------------------------------
def load_class_names():
    dataset = load_from_disk("/data/CPE_487-587/imagenet-1k-arrow")
    return dataset["train"].features["label"].names


# -------------------------------------------------
# Preprocessing (must match training)
# -------------------------------------------------
def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])


# -------------------------------------------------
# Inference
# -------------------------------------------------
def main():

    args = parse_args()

    # Load ONNX model
    session = ort.InferenceSession(args.model)

    # Load class names
    class_names = load_class_names()

    # Load image
    image = Image.open(args.image).convert("RGB")

    transform = get_transform()
    input_tensor = transform(image).unsqueeze(0)  # (1,3,224,224)

    # Convert to numpy
    input_numpy = input_tensor.numpy()

    # Run inference
    outputs = session.run(None, {"input": input_numpy})
    logits = outputs[0]

    # Convert to probabilities
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

    # Top-1
    top1_idx = np.argmax(probs)
    top1_prob = probs[0][top1_idx]

    print("\n===== Top-1 Prediction =====")
    print(f"Class ID: {top1_idx}")
    print(f"Class Name: {class_names[top1_idx]}")
    print(f"Confidence: {top1_prob:.4f}")

    # Top-5
    top5_idx = np.argsort(probs[0])[-5:][::-1]

    print("\n===== Top-5 Predictions =====")
    for i in top5_idx:
        print(f"{class_names[i]} : {probs[0][i]:.4f}")


# -------------------------------------------------
if __name__ == "__main__":
    main()
