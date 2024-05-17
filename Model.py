import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    setup = [19, 3, 32, 2]
    
    def __init__(self, in_size, out_size, hidden_size, n_layers, lr=0.003):
        super().__init__()
        self.fc1 = nn.Linear(in_size, hidden_size)
        self.fcx = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(n_layers)]) # this is a list of linear layers
        self.fc2 = nn.Linear(hidden_size, out_size)
        self.lr = lr
    
    def forward(self, inputs):
        x = F.gelu(self.fc1(inputs))
        for hidden in self.fcx: # iterating over hidden layers
            x = F.gelu(hidden(x))  # applying each hidden layer
        return torch.softmax(self.fc2(x), axis=-1)

    def get_model_copy(self):
        state = self.state_dict()
        newmodel = MLP(*MLP.setup, self.lr*0.8)
        newmodel.load_state_dict(state)
        return newmodel

    def train(self, enviroment, actions, past_interaction=50): 
        past_interaction *= 1

        buttons = actions
        radar = enviroment

        # crashed_timestamp = radar.argmin(-1) if (radar[radar.argmin(-1)].sum() == 0) else -1 
        # context = 1000
        # buttons = buttons[crashed_timestamp-context:crashed_timestamp]
        # radar = radar[crashed_timestamp-context:crashed_timestamp]
        buttons[-past_interaction:] = 1-buttons[-past_interaction:]
        buttons = buttons[-2*past_interaction:]
        radar = radar[-2*past_interaction:]

        # print(buttons)

        # train model   
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.8)
        from tqdm import tqdm
        for epoch in (range(10)):
            optimizer.zero_grad()

            output = self(radar)
            loss = F.mse_loss(output, buttons)
            loss.backward()
            optimizer.step()
            # p.set_description(f"Loss: {loss.item():2f} at epoch {epoch:2d}")
        
        state = self.state_dict()
        newmodel = MLP(*MLP.setup, self.lr)
        newmodel.load_state_dict(state)
        return newmodel
