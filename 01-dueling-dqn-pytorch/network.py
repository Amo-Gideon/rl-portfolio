import torch
import torch.nn as nn


class DuelingDQN(nn.Module):
    """
    Dueling DQN from Wang et al. 2016, Equation (9).
    Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
    """

    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.n_actions = n_actions

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            conv_out = self.conv(dummy)
            self.conv_out_size = conv_out.view(1,-1).size(1)
        

        self.fc = nn.Sequential(
            nn.Linear(self.conv_out_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )
        # Two streams: Value (scalar) and Advantage (vector)
        self.value_stream = nn.Linear(512, 1)
        self.advantage_stream = nn.Linear(512, n_actions)

       

        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        value = self.value_stream(x)
        advantages = self.advantage_stream(x)

        mean_advantage = advantages.mean(dim=1, keepdim=True)
        q_values = value + (advantages - mean_advantage)
        return q_values

     

class StandardDQN(nn.Module):
    """
    Standard single-stream Q-network.
    Same conv + FC backbone as DuelingDQN, but one output head.
    """
    def __init__(self, input_shape, n_actions):
        super().__init__()

        self.conv = nn.Sequential(
           nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
           nn.ReLU(),
           nn.Conv2d(32, 64, kernel_size=4, stride=2),
           nn.ReLU(),
           nn.Conv2d(64, 64, kernel_size=3, stride=1),
           nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            conv_out = self.conv(dummy)
            self.conv_out_size = conv_out.view(1,-1).size(1)
        
        self.fc = nn.Sequential(
            nn.Linear(self.conv_out_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        q_values = self.fc(x)
        return q_values

    
if __name__ == "__main__":
    # Atari-like input: 4 stacked frames, 84 * 84
    input_shape = (4, 84, 84)
    n_actions = 6

    # Test Dueling
    dueling_net = DuelingDQN(input_shape, n_actions)
    x = torch.randn(2, *input_shape)
    out = dueling_net(x)
    print(f"DuelingDQN output shape: {out.shape}")

    # Test Standard
    standard_net = StandardDQN(input_shape, n_actions)
    out2 = standard_net(x)
    print(f"StandardDQN output shape: {out2.shape}")

    dueling_params = sum(p.numel() for p in dueling_net.parameters())
    standard_params = sum(p.numel() for p in standard_net.parameters())
    print(f"Dueling params: {dueling_params:,}")
    print(f"Standard params: {standard_params:,}")
    print(f"Diff: {abs(dueling_params - standard_params):,}")





