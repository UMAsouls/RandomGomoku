import numpy as np
import os
import torch
from alpha_gomoku import AlphaGomokuAgent
import GomokuEnv

def augment_data(states, policies, values):
    """Augment training data with rotations and reflections"""
    augmented_states = []
    augmented_policies = []
    augmented_values = []
    
    for state, policy, value in zip(states, policies, values):
        # Original
        augmented_states.append(state)
        augmented_policies.append(policy)
        augmented_values.append(value)
        
        # 90 degree rotation
        rotated_state = np.rot90(state, k=1, axes=(1, 2))
        rotated_policy = np.rot90(policy.reshape(19, 19), k=1).flatten()
        augmented_states.append(rotated_state)
        augmented_policies.append(rotated_policy)
        augmented_values.append(value)
        
        # 180 degree rotation
        rotated_state = np.rot90(state, k=2, axes=(1, 2))
        rotated_policy = np.rot90(policy.reshape(19, 19), k=2).flatten()
        augmented_states.append(rotated_state)
        augmented_policies.append(rotated_policy)
        augmented_values.append(value)
        
        # 270 degree rotation
        rotated_state = np.rot90(state, k=3, axes=(1, 2))
        rotated_policy = np.rot90(policy.reshape(19, 19), k=3).flatten()
        augmented_states.append(rotated_state)
        augmented_policies.append(rotated_policy)
        augmented_values.append(value)
    
    return augmented_states, augmented_policies, augmented_values

def self_play(agent, num_games=10):
    """Generate training data through self-play"""
    training_states = []
    training_policies = []
    training_values = []
    
    for game_idx in range(num_games):
        print(f"Playing game {game_idx+1}/{num_games}")
        env = GomokuEnv.GomokuEnv()
        state = env.board.copy()
        done = False
        game_history = []
        
        while not done:
            current_player = env.current_player
            
            # Store the current state
            encoded_state = agent.encode_state(state, current_player)
            
            # Get action probabilities from MCTS
            root = agent.get_action(state, current_player)
            
            # Store the MCTS policy (based on visit counts)
            policy = np.zeros(agent.board_size * agent.board_size)
            for child in root.children:
                if child.action:
                    x, y = child.action
                    policy[y * agent.board_size + x] = child.visits
            
            # Normalize the policy
            if policy.sum() > 0:
                policy = policy / policy.sum()
            
            # Store state and policy for training
            game_history.append({
                'state': encoded_state,
                'policy': policy,
                'player': current_player
            })
            
            # Take the action
            action = root.action
            next_state, reward, done, _ = env.step(action)
            state = next_state.copy()
        
        # Calculate game result
        winner = env.winner
        
        # Assign values based on game outcome
        for history in game_history:
            player = history['player']
            if winner == 0:  # Draw
                value = 0.0
            elif winner == player:  # Win
                value = 1.0
            else:  # Loss
                value = -1.0
                
            training_states.append(history['state'])
            training_policies.append(history['policy'])
            training_values.append(value)
    
    # Augment the training data
    training_states, training_policies, training_values = augment_data(
        training_states, training_policies, training_values)
    
    return training_states, training_policies, training_values

def train_agent(iterations=50, games_per_iteration=10, batch_size=128, epochs=10, model_dir="models"):
    """Main training loop"""
    board_size = 19
    
    # # Find the latest model to continue training
    # latest_model_path = None
    # if os.path.exists(model_dir):
    #     model_files = [f for f in os.listdir(model_dir) if f.startswith("alpha_gomoku_iter_") and f.endswith(".pt")]
    #     if model_files:
    #         latest_model_file = max(model_files, key=lambda f: int(f.split('_')[-1].split('.')[0]))
    #         latest_model_path = os.path.join(model_dir, latest_model_file)
    
    # agent = AlphaGomokuAgent(board_size=board_size, model_path=latest_model_path)
    agent = AlphaGomokuAgent(board_size=board_size)
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    for iteration in range(iterations):
        print(f"Starting iteration {iteration+1}/{iterations}")
        
        # Self-play to generate training data
        states, policies, values = self_play(agent, num_games=games_per_iteration)
        
        print(f"Training on {len(states)} examples")
        
        # Train the neural network
        agent.train(states, policies, values, batch_size=batch_size, epochs=epochs)
        
        # Save the model
        agent.save_model(os.path.join(model_dir, f"alpha_gomoku_iter_{iteration+1}.pt"))
    
    # Save the final model
    agent.save_model(os.path.join(model_dir, "alpha_gomoku_final.pt"))

if __name__ == "__main__":
    train_agent()
