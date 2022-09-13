from ctypes import Union
from torchvision import transforms, models
from transformers import ViTFeatureExtractor, ViTForImageClassification
import torch
from PIL import Image

model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
model.eval()

def preprocess_image(path) -> torch.Tensor:
    """
    Function to open and preprocess the image to give it to the desenet model

    input : str → path to the image
    output : inout_batch → Tensor
    """
    input_img = Image.open(path)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_img)
    input_batch = input_tensor.unsqueeze(0)
    return  input_batch

def get_prediction(input_image) -> Union[str, int]:
    """
    Function to predict via the densenet model what is in the image
    input : inputs batch
    output : category_id → int, categorie → str
    """
    with torch.no_grad():
        output = model(input_image)


    print(output[0])
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    print(probabilities)



    # Read the categories
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    # Show top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())

    return categories[top5_catid[0]], top5_catid[0].item()


feature_extractor_vit = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-384')
model_vit = ViTForImageClassification.from_pretrained('google/vit-base-patch16-384')

def image_classification_vit(path) -> Union[str, int]:
    """
    Functio to use the vit model to do image classification
    input : str → path to the image
    returns res → str class of the image, predicted class id → int
    """
    inputs = feature_extractor_vit(images=Image.open(path), return_tensors="pt") #preprossesing
    outputs = model_vit(**inputs)
    logits = outputs.logits
    # model predicts one of the 1000 ImageNet classes
    predicted_class_idx = logits.argmax(-1).item()
    res = model_vit.config.id2label[predicted_class_idx].split(",")[0]
    return res, predicted_class_idx
