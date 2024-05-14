import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    setup = [19, 3, 16, 2]
    
    def __init__(self, in_size, out_size, hidden_size, n_layers):
        super().__init__()
        self.fc1 = nn.Linear(in_size, hidden_size)
        self.fcx = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(n_layers)]) # this is a list of linear layers
        self.fc2 = nn.Linear(hidden_size, out_size)
    
    def forward(self, inputs):
        x = F.gelu(self.fc1(inputs))
        for hidden in self.fcx:    # iterating over hidden layers
            x = F.gelu(hidden(x))  # applying each hidden layer
        return torch.softmax(self.fc2(x), axis=-1)

    def train(self, past_interaction = 10, epochs = 50, recording=None): 
        past_interaction *= 1
        bestmodel = self

        # make labels so that buttons that lead to high speed 20 steps after are rewarded
        buttons = recording[:, -3:]
        radar = recording[:, :MLP.setup[0]]

        crashed_timestamp = radar.argmin(-1) if (radar[radar.argmin(-1)].sum() == 0) else -1 
        context = 250
        buttons = buttons[crashed_timestamp-context:crashed_timestamp]
        radar = radar[crashed_timestamp-context:crashed_timestamp]

        buttons[:-past_interaction] = 1-buttons[:-past_interaction]

        # train model   
        optimizer = torch.optim.SGD(bestmodel.parameters(), lr=0.0003)
        from tqdm import tqdm
        for epoch in (range(epochs)):
            optimizer.zero_grad()

            output = bestmodel(radar)
            loss = F.mse_loss(output, buttons)
            loss.backward()
            optimizer.step()
            # p.set_description(f"Loss: {loss.item():2f} at epoch {epoch:2d}")
        
        state = bestmodel.state_dict()
        newmodel = MLP(*MLP.setup)
        newmodel.load_state_dict(state)
        return newmodel
