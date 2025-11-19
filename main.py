import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

class SimpleWorldModel(nn.Module):
    """
    A simple world model that learns ball physics from visual data (2D positions)
    """
    def __init__(self, input_dim=2, hidden_dim=64, latent_dim=8):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        self.transition = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, observation, action):
        latent_state = self.encoder(observation)
        
        next_latent = self.transition(torch.cat([latent_state, action], dim=1))
        
        predicted_obs = self.decoder(next_latent)
        
        return predicted_obs, latent_state, next_latent

def generate_ball_data(num_samples=1000):
    """
    Generate training data: ball positions with simple physics
    Action = force applied, affects velocity
    """
    observations = []
    actions = []
    next_observations = []
    
    for i in range(num_samples):
        pos = np.random.uniform(-5, 5, 2)
        vel = np.random.uniform(-1, 1, 2)
        
        action = np.random.uniform(-2, 2, 1)
        action_direction = np.random.uniform(-1, 1, 2)
        action_direction = action_direction / (np.linalg.norm(action_direction) + 1e-8)
        
        new_vel = vel + action * action_direction * 0.1
        new_pos = pos + new_vel
        
        new_pos += np.random.normal(0, 0.1, 2)
        
        observations.append(pos)
        actions.append(action)
        next_observations.append(new_pos)
    
    return (torch.FloatTensor(observations),
            torch.FloatTensor(actions),
            torch.FloatTensor(next_observations))

def train_world_model():
    """Train our simple world model"""
    observations, actions, next_observations = generate_ball_data(2000)

    model = SimpleWorldModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    losses = []
    for epoch in range(500):
        optimizer.zero_grad()
        
        predicted_next, _, _ = model(observations, actions)
        
        loss = criterion(predicted_next, next_observations)
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    return model, losses

def test_world_model(model):
    """Test the trained world model with a trajectory prediction"""
    start_pos = torch.FloatTensor([[0.0, 0.0]])
    current_pos = start_pos.clone()
    
    actions = [torch.FloatTensor([[1.0]]),  
               torch.FloatTensor([[0.5]]),  
               torch.FloatTensor([[-1.0]]), 
               torch.FloatTensor([[0.0]])]  
    
    trajectory = [current_pos.detach().numpy()[0]]
    predicted_trajectory = [current_pos.detach().numpy()[0]]
    
    print("\nTesting World Model Prediction:")
    print("Step 0: Position [0.0, 0.0]")
    
    for i, action in enumerate(actions):
        with torch.no_grad():
            predicted_next, _, _ = model(current_pos, action)
        
        current_pos_np = current_pos.numpy()[0]
        action_val = action.numpy()[0][0]
        
        true_next = current_pos_np + np.array([action_val * 0.2, action_val * 0.1])
        
        print(f"Step {i+1}: Action {action_val:.1f}")
        print(f"  World Model Prediction: {predicted_next.numpy()[0]}")
        print(f"  True Physics: {true_next}")
        
        current_pos = torch.FloatTensor([true_next])
        trajectory.append(true_next)
        predicted_trajectory.append(predicted_next.numpy()[0])
    
    return trajectory, predicted_trajectory

def plot_results(trajectory, predicted_trajectory, losses):
    """Plot the training results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    trajectory = np.array(trajectory)
    predicted_trajectory = np.array(predicted_trajectory)
    
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'bo-', label='True Path', linewidth=2)
    ax1.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], 'ro--', label='Predicted Path', linewidth=2)
    ax1.scatter(trajectory[0, 0], trajectory[0, 1], color='green', s=100, label='Start', zorder=5)
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_title('World Model Prediction vs True Physics')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(losses)
    ax2.set_yscale('log')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss (log scale)')
    ax2.set_title('Training Loss')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("=== Simple World Model Demo ===")
    print("Training a world model to predict ball physics...")
    
    model, losses = train_world_model()

    trajectory, predicted_trajectory = test_world_model(model)
    
    plot_results(trajectory, predicted_trajectory, losses)
    
    print("\n=== Model Architecture ===")
    print(model)
    
    print("\n=== Planning Demo ===")
    print("Finding action sequence to reach target position...")
    
    start_pos = torch.FloatTensor([[0.0, 0.0]])
    target_pos = torch.FloatTensor([[2.0, 1.0]])
    
    best_action = None
    best_distance = float('inf')
    
    for test_action in torch.linspace(-2, 2, 20):
        with torch.no_grad():
            predicted_next, _, _ = model(start_pos, test_action.unsqueeze(0).unsqueeze(1))
        
        distance = torch.norm(predicted_next - target_pos)
        
        if distance < best_distance:
            best_distance = distance
            best_action = test_action.item()
    
    print(f"Best action to reach target: {best_action:.2f}")
    print(f"Predicted position after action: {model(start_pos, torch.FloatTensor([[best_action]]))[0].numpy()[0]}")
    print(f"Target position: {target_pos.numpy()[0]}")