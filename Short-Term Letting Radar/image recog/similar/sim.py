import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

def generate_embedding(model, preprocess, image_path):
    image = Image.open(image_path)
    image = preprocess(image)
    image = image.unsqueeze(0)

    with torch.no_grad():
        embeddings = model(image)
    
    return embeddings.squeeze(0)

def cosine_similarity(embedding1, embedding2):
    return torch.nn.functional.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0))

def find_most_similar_imgs(query_embedding, all_embeddings, image_paths, top_k=5):
    similarities = [(cosine_similarity(query_embedding, emb), idx) for idx, emb in enumerate(all_embeddings)]
    similarities.sort(key=lambda x: x[0], reverse=True)
    top_indices = similarities[:top_k]
    return [(image_paths[idx], idx) for _, idx in top_indices]

def main():
    model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_paths = ["./images/image_1.jpg", "./images/image_2.jpg", "./images/image_3.jpg", "./images/image_4.jpg", "./images/image_5.jpg"]
    all_embeddings = [generate_embedding(model, preprocess, path) for path in image_paths]

    query_image_path = "./images/query.jpg"
    query_embedding = generate_embedding(model, preprocess, query_image_path)

    similar_images = find_most_similar_imgs(query_embedding, all_embeddings, image_paths, top_k=5)
    print(f"Most similar images: {similar_images}")

if __name__ == "__main__":
    main()