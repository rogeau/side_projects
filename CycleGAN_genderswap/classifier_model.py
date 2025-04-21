import torch.nn as nn
import torch

class GenderClassifier(nn.Module):
    def __init__(self, n_chan=3, n_dim=(256, 256)):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(n_chan, 32, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(n_dim[0]//8 * n_dim[0]//8 * 128, 512), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

def test():
    x = torch.randn((2, 3, 256, 256))
    classifier= GenderClassifier(n_chan=3, n_dim=(256, 256))
    preds = classifier(x)
    print(preds.shape)
    print(classifier)

if __name__ == "__main__":
    test()