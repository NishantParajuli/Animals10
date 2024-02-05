from django.shortcuts import render
from django.conf import settings
from .forms import ImageUploadForm

import base64
import io
import os
from torchvision import transforms, models
from torchvision.transforms import v2

from torch import load, uint8, float32, softmax, argmax
from PIL import Image

# load pretrained EfficientNet and go straight to evaluation mode for inference
# load as global variable here, to avoid expensive reloads with each request
model = models.efficientnet_v2_m(num_classes=10)
model.load_state_dict(
    load(os.path.join(settings.STATIC_ROOT, 'home\AnimalClassifierEffNetV2-M.pth')))
model.eval()

class_names = ['butterfly',
               'cat',
               'chicken',
               'cow',
               'dog',
               'elephant',
               'horse',
               'sheep',
               'spider',
               'squirrel']


def transform_image(image_bytes):
    # define transforms to be applied to input image
    transform = transforms.Compose([
        transforms.v2.ToImage(),
        transforms.v2.ToDtype(uint8, scale=True),
        transforms.v2.Resize((256, 256), antialias=True),
        transforms.v2.CenterCrop(224),
        transforms.v2.ToDtype(float32, scale=True),
        transforms.v2.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
    ])
    # convert image from bytes to PIL Image
    image = Image.open(io.BytesIO(image_bytes))
    # apply transforms
    return transform(image).unsqueeze(dim=0)


def get_prediction(image_bytes):
    # convert image to tensor
    tensor = transform_image(image_bytes=image_bytes)
    # get prediction
    outputs = model(tensor)
    # get class probabilities
    probs = softmax(outputs, dim=1)
    # get predicted class
    predicted = argmax(probs, dim=1)
    # return predicted class
    return class_names[predicted], probs[0][predicted].item()


def index(request):
    image_uri = None
    predicted_class = None
    predicted_probability = 0.0

    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            image_bytes = image.file.read()
            # convert and pass the image as base64 string to avoid storing it to DB or filesystem
            encoded_img = base64.b64encode(image_bytes).decode('ascii')
            image_uri = 'data:%s;base64,%s' % ('image/jpeg', encoded_img)

            try:
                predicted_class, predicted_probability = get_prediction(
                    image_bytes=image_bytes)
            except RuntimeError as re:
                print(re)
                predicted_class = 'Error'
                predicted_probability = 0.0

    else:
        form = ImageUploadForm()

    context = {
        "form": form,
        'image_uri': image_uri,
        'predicted_class': predicted_class,
        'predicted_probability': round(predicted_probability * 100, 2)
    }
    return render(request, 'home/index.html', context)
