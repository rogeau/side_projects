from torch.utils.data import random_split, DataLoader
from dataset import Flickr8kDataset, val_transform, train_transform, detokenize
from torchvision.models import resnet50, ResNet50_Weights
from resnet import TruncatedResNet
from transformer import Decoder
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import itertools
import json
import configs

torch.manual_seed(21)

with open('codebook.json') as f:
    codebook = json.load(f)

full_dataset = Flickr8kDataset()
full_dataset.codebook = codebook

train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_dataset.dataset.transform = train_transform()
val_dataset.dataset.transform = val_transform()

pretrained_whole_resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
cnn = TruncatedResNet(pretrained_whole_resnet).to(configs.DEVICE)

transformer = Decoder(vocab_size=len(codebook)).to(configs.DEVICE)

optimizer = optim.Adam(
    list(cnn.parameters()) + list(transformer.parameters()), 
    lr=1e-4, 
    betas=(0.9, 0.999)
)


for epoch in range(configs.EPOCHS):

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    #train_loader = itertools.islice(train_loader, 5)
    val_loader  = DataLoader(val_dataset, batch_size=32, shuffle=False)
    #val_loader = itertools.islice(val_loader, 5)

    cnn.eval()
    transformer.eval()

    val_loss = 0
    val_tokens = 0

    with torch.no_grad():
        image, caption, mask, idx = next(iter(val_loader))
        image, caption, mask = image.to(configs.DEVICE), caption.to(configs.DEVICE), mask.to(configs.DEVICE)
        features = cnn(image)
        batch_size, seq_len = caption.size()
        input_seq = torch.full((batch_size, 1), codebook['[SOS]'], dtype=torch.long, device=configs.DEVICE)

        for t in range(1, seq_len):
            logits, _ = transformer(input_seq, features)
            next_token_logits = logits[:, -1, :]

            next_token_probs = F.softmax(next_token_logits, dim=-1)
            predicted = torch.multinomial(next_token_probs, num_samples=1)
            input_seq = torch.cat([input_seq, predicted], dim=1)

        original_input = full_dataset.image_captions[idx[-1]]
        print(original_input)
        pred_sentence = detokenize(input_seq[-1, :], codebook)
        print(input_seq[-1, :])
        print(f'\nPredicted sentence:\n{pred_sentence}')
        target_sentence = detokenize(caption[-1, :], codebook)
        print(f'Target sentence:\n{target_sentence}\n')
        

    cnn.train()
    transformer.train()

    train_loss = 0

    loop_train = tqdm(train_loader, desc=f'Train epoch {epoch+1}/{configs.EPOCHS}', leave=True)
    loop_val = tqdm(val_loader, desc=f'Val epoch {epoch+1}/{configs.EPOCHS}', leave=True)

    for image, caption, mask, _ in loop_train:
        image, caption, mask = image.to(configs.DEVICE), caption.to(configs.DEVICE), mask.to(configs.DEVICE)
        optimizer.zero_grad()

        caption_input = caption[:, :-1]
        caption_target = caption[:, 1:]
        mask_input = mask[:, :-1]
        mask_target = mask[:, 1:]

        features = cnn(image)
        logits, _ = transformer(caption_input, features)
        
        logits = logits.view(-1, logits.size(-1))
        caption_target = caption_target.contiguous().view(-1)
        mask_target = mask_target.contiguous().view(-1)

        loss = F.cross_entropy(logits, caption_target, reduction='none')
        loss = (loss * mask_target).sum() / mask_target.sum()

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f"Epoch {epoch+1}: Training loss = {train_loss:.3f}")


torch.save(cnn.state_dict(), 'cnn.pth')
torch.save(transformer.state_dict(), 'transformer.pth')

        # for image, caption, mask in loop_val:
        #     image, caption, mask = image.to(configs.DEVICE), caption.to(configs.DEVICE), mask.to(configs.DEVICE)
        #     features = cnn(image)
        #     batch_size, seq_len = caption.size()
        #     input_seq = torch.full((batch_size, 1), codebook['[SOS]'], dtype=torch.long, configs.DEVICE=configs.DEVICE)

        #     for t in range(1, seq_len):
        #         logits, _ = transformer(input_seq, features)
        #         next_token_logits = logits[:, -1, :]

        #         target = caption[:, t]
        #         current_mask = mask[:, t]
        #         loss = F.cross_entropy(next_token_logits, target, reduction='none') 
        #         val_loss += (loss * current_mask).sum().item()
        #         val_tokens += current_mask.sum().item()

        #         next_token_probs = F.softmax(next_token_logits, dim=-1)
        #         predicted = torch.multinomial(next_token_probs, num_samples=1)
        #         input_seq = torch.cat([input_seq, predicted], dim=1)

        # pred_sentence = detokenize(input_seq[-1, :], codebook)
        # print(f'\nPredicted sentence:\n{pred_sentence}')
        # target_sentence = detokenize(caption[-1, :], codebook)
        # print(f'Target sentence:\n{target_sentence}\n')

    # val_loss /= val_tokens
    # print(f"Epoch {epoch+1}: Autoregressive Validation Loss = {val_loss:.3f}")