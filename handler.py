import unzip_requirements
from io import BytesIO
import base64
import boto3
import json
import torch
import transforms
from PIL import Image

data = BytesIO()
s3 = boto3.client("s3")

bucket = "your-bucket-name"
model_key = "your-model-name.pt"

s3.download_fileobj(bucket, model_key, data)

model_file = BytesIO(data.getvalue())

model = torch.load(model_file)

with open("imagenet_classes.txt") as f:
    labels = [line.strip() for line in f.readlines()]

transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def inference(event, context):
    if "source" in event and event["source"] == "serverless-plugin-warmup":
        print("Candidate Image Analysis Lambda is warm!")
        return None

    if event["httpMethod"] != "POST":
        return

    body = {
        "message": "Go Serverless v1.0! Your function executed successfully!",
        "input": event,
    }

    data = json.loads(event["body"])
    name = data["name"]
    image = data["file"]
    image = image[image.find(",") + 1 :]
    dec = base64.b64decode(image + "===")
    img = Image.open(dec)

    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)

    out = model(batch_t)
    _, index = torch.max(out, 1)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

    _, indices = torch.sort(out, descending=True)
    [(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]

    print(labels[index[0]], percentage[index[0]].item())

    response = {
        "statusCode": 200,
        "body": json.dumps(
            [(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]
        ),
    }

    return response
