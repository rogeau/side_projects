from dataset import Flickr8kDataset, build_vocab
import json

full_dataset = Flickr8kDataset()
codebook = build_vocab(full_dataset)

with open('codebook.json', 'w') as f:
    json.dump(codebook, f)