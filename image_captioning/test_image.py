from PIL import Image
import torch
import torch.nn.functional as F
import json
from dataset import detokenize, val_transform
from resnet import TruncatedResNet
from transformer import Decoder
from torchvision.models import resnet50
import configs

with open('codebook.json') as f:
    codebook = json.load(f)

def test_on_image(img_path, transform, cnn, transformer, codebook):
    image = Image.open(img_path).convert("RGB")

    transform
    tensor_image = transform(image).unsqueeze(0).to(configs.DEVICE)

    features = cnn(tensor_image)

    input_seq = torch.full((1, 1), codebook['[SOS]'], dtype=torch.long, device=configs.DEVICE)

    for t in range(1, 50):
        logits, _ = transformer(input_seq, features)
        next_token_logits = logits[:, -1, :]

        next_token_probs = F.softmax(next_token_logits, dim=-1)
        predicted = torch.multinomial(next_token_probs, num_samples=1)
        input_seq = torch.cat([input_seq, predicted], dim=1)

    pred_sentence = detokenize(input_seq[-1, :], codebook)
    print(input_seq[-1, :])
    print(f'\nPredicted sentence:\n{pred_sentence}')

def main():
    cnn = TruncatedResNet(resnet50()).to(configs.DEVICE)
    cnn.load_state_dict(torch.load('cnn.pth'))
    transformer = Decoder(vocab_size=len(codebook)).to(configs.DEVICE)
    transformer.load_state_dict(torch.load('transformer.pth'))

    img_path = "test_img/IMG_4651.jpg"

    test_on_image(img_path=img_path, transform=val_transform(), cnn=cnn, transformer=transformer, codebook=codebook)

if __name__ == "__main__":
    main()