# Reinforcement Learning (RL) Challenge - Data Chefs @ TIL-25 Hackathon

![RL Agent Preview](README%20Preview.gif)

This repository (`lolkabash/til-25-data-chefs-RL`) contains the code, models, and notebooks developed by the **Data Chefs** team for the Reinforcement Learning (RL) challenge of the DSTA BrainHack TIL-AI 2025. Our primary focus was on training an RL agent for autonomous navigation in the competition environment.

The work here primarily consists of Jupyter Notebooks, reflecting our iterative process of model development, training, and evaluation.

## 📝 Description

The DSTA BrainHack TIL-AI 2025 RL challenge involved developing algorithms for a simulated robot to autonomously navigate an environment through trial-and-error learning. The environment featured a maze layout where teams competed as "Scout" or "Guard" in a round-robin format over four rounds.

## 🔗 Repository Link

*   **This Repository:** [lolkabash/til-25-data-chefs-RL](https://github.com/lolkabash/til-25-data-chefs-RL)
*   **Main Team Repository:** The overall project and other challenges were managed in our main team repository: [lolkabash/til-25-data-chefs](https://github.com/lolkabash/til-25-data-chefs)

## 💻 Key Technologies We Used

*   **Python:** Primary programming language for implementing the RL algorithms.
*   **Jupyter Notebooks:** Used extensively for model development, training, and evaluation.
*   **PyTorch:** The deep learning framework used for building and training the neural networks.
*   **Deep Q-Network (DQN):** The core RL algorithm. The Q-function was approximated using a Multi-Layer Perceptron (MLP) processing engineered visual and state features.
*   **Prioritized Experience Replay (PER):** Implemented using a SumTree to sample more important transitions more frequently.
*   **Epsilon-Greedy Exploration:** Used for action selection during training, with a decaying epsilon value.
*   **Target Network & Double DQN:** Utilized to stabilize training and reduce Q-value overestimation.
*   **Adam Optimizer:** Used for training the neural network.
*   **Custom Reward Shaping:** Extensive, fine-grained reward structure to guide agent behavior.
*   **`til_environment.gridworld`:** The custom game environment (from the `til-25-environment` submodule provided by the competition), adhering to a PettingZoo-like API for multi-agent interactions.

## ✨ Our Solution & Key Achievements
Our approach to the RL challenge was highly iterative, as reflected in the various experimental "Attempts" and notebooks within this repository (e.g., "Attempt 1 MARL PPO", "Attempt 2 MLP No Reward", "Attempt 5 CNN DQN"). This process of experimentation with different architectures (MLP, CNN concepts), reward structures, and training methodologies culminated in our final model, documented as "Attempt 10 CNN DQN With Checkpoints."

The core of this final agent was a Deep Q-Network (DQN) with the following key characteristics:

*   **Model Architecture:** The DQN employed a Multi-Layer Perceptron (MLP) with multiple hidden layers (256 units each) to approximate Q-values. While named "CNN DQN" in our attempts (referring to the visual nature of the input), the network processed a 288-dimensional feature vector carefully engineered from the game state, rather than raw pixels. This feature vector included:
    *   A processed representation of the agent's 7x5 viewcone (8 features per tile).
    *   One-hot encoded agent direction.
    *   Normalized agent location (x, y coordinates).
    *   A binary indicator for the Scout role.
    *   The normalized current step count within the episode.
*   **Advanced Training Techniques:**
    *   **Prioritized Experience Replay (PER):** We implemented PER (using a SumTree) to enable the agent to learn more efficiently by focusing on surprising or significant experiences.
    *   **Double DQN:** To reduce overestimation of Q-values and improve stability, the target Q-values were calculated using the policy network to select the best next action and the target network to evaluate that action.
    *   **Epsilon-Greedy Exploration:** An epsilon-greedy strategy was used for action selection during training, with epsilon decaying from an initial value of 0.25 down to 0.01.
    *   **Resumable Training with Checkpoints:** The agent was designed to save and load model checkpoints (`.pth` files). This was crucial for enabling long-duration training sessions that could be resumed, facilitating continuous improvement throughout the hackathon (as seen in "Attempt 9 CNN DQN Resume Training" and "Attempt 10 CNN DQN With Checkpoints").
*   **Customized Reward System (`CUSTOM_REWARDS_DICT`):** A detailed reward structure was critical for guiding agent behavior. This included:
    *   Significant positive rewards for achieving Scout objectives (mission completion, recon, survival).
    *   Large negative penalties for critical failures (Scout captured, wall collisions, agent collisions).
    *   Small penalties per step (time penalty) and for stationary behavior to encourage efficient movement. This tailored reward system (explored in attempts like "Attempt 6 Agent 03 Reward") allowed for differentiated training signals for agents playing as Scout or Guard.
*   **Iterative Refinement:** The final model was the product of extensive experimentation. Early attempts explored concepts like MARL PPO ("Attempt 1") and simpler MLP/DQN versions ("Attempt 2", "Attempt 4", "Attempt 5"). We also experimented with different agent training strategies ("Attempt 7 CNN DQN Train All Agents Simultaneously", "Attempt 8 CNN DQN Train Seperate Agents") and reward designs ("Attempt 3 Guard DQNV2") before arriving at the configuration in "Attempt 10". This iterative process allowed us to fine-tune hyperparameters (learning rate, buffer size, update frequencies) and the overall agent design for optimal performance.
