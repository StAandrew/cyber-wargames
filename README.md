# Cyber War Games

**Multi-Agent Reinforcement Learning for Denial-of-Service Attacks**

UCL Department of Electronic & Electrical Engineering | 3rd Year Project | Andrey Staniukynas | Supervisor: Prof Rio

---

## Overview

This project simulates Denial-of-Service (DoS) attacks using two competing reinforcement learning agents in a network environment. An **Attacker** agent learns to craft evasive malicious packets to flood a target, while a **Defender** agent learns to distinguish attack traffic from legitimate background traffic. Both agents train simultaneously in an adversarial setup, inspired by OpenAI's multi-agent hide-and-seek research.

The network is simulated locally: all traffic flows through a central **Router**, with a **Background Traffic** generator producing legitimate packets. The Defender must allow legitimate traffic and block attacks; the Attacker must fool the Defender into accepting its packets.

```
                           legitimate
  Background Traffic  ───────packets───────>  Router  <───malicious───  Attacker
                                                |          packets
                                           all packets
                                                |
                                                v
                                            Defender
                                           /        \
                                    allow /          \ drop
                                         v            v
                                      Server        (discarded)
```

## Agents

### Defender (DQN)

- **Algorithm:** Deep Q-Network (DQN) via Stable Baselines 3
- **Action space:** Discrete(2) -- `0` = allow packet, `1` = deny packet
- **Observation:** `[source_ip, time_delta]` -- the packet's source IP and time since last packet
- **Reward:** +1 for correct classification (true positive / true negative), -1 for misclassification (false positive / false negative)

### Attacker (TD3)

- **Algorithm:** Twin Delayed DDPG (TD3) via Stable Baselines 3
- **Action space:** Box(0, 500, shape=(2,)) -- continuous
  - `action[0]`: spoofed source IP address
  - `action[1]`: delay before sending the next packet
- **Observation:** `[reward]` -- whether the previous packet was allowed or denied
- **Reward:** +1 if the Defender allowed the packet (successful attack), -1 if denied

### Difficulty Levels

The agents can learn at increasing levels of difficulty:

| Level | Features available to agents |
|-------|----------------------------|
| 1 | IP address only |
| 2 | IP address + packet timing |
| 3 | IP address + timing + packet contents/headers |

The project reached Level 2. The Defender learned to distinguish traffic by IP address within ~15 minutes (66% accuracy) and reached >93% accuracy within an hour.

## Project Structure

```
.
├── router.py          # TCP packet router -- forwards traffic between agents
├── env_def.py         # OpenAI Gym environment for the Defender (DQN)
├── env_atk.py         # OpenAI Gym environment for the Attacker (TD3)
├── bg_traffic.py      # Background traffic generator (legitimate packets)
├── common.py          # NetworkPacket class and send_and_receive helper
├── config.py          # Global config: host, port, packet size, logging
├── train_def.py       # Train the Defender agent
├── train_atk.py       # Train the Attacker agent
├── play_def.py        # Evaluate a trained Defender model
├── play_atk.py        # Evaluate a trained Attacker model
├── env_check.py       # Utility to validate Gym environments
├── cleanup.py         # Utility to clear models/ and logs/ directories
├── requirements.txt   # Python dependencies
├── models/            # Saved model checkpoints (.zip)
└── logs/              # TensorBoard training logs
```

## Requirements

- Python 3.9+
- macOS (developed and tested on macOS; not tested on Windows)

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/StAandrew/cyber-wargames.git
cd cyber-wargames

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Usage

### Evaluating trained models

Run each command in a **separate terminal**, in this order:

```bash
# Terminal 1 -- TensorBoard (optional, for viewing training metrics)
tensorboard --logdir=logs

# Terminal 2 -- Start the router (must be first)
python3 router.py

# Terminal 3 -- Start the Defender
python3 play_def.py

# Terminal 4 -- Start the Attacker
python3 play_atk.py
```

The Defender must connect before the Attacker. The router must be running before either agent.

### Training from scratch

Replace `play_def.py` and `play_atk.py` with the training scripts:

```bash
# Terminal 2 -- Router
python3 router.py

# Terminal 3 -- Train Defender (spawns background traffic automatically)
python3 train_def.py

# Terminal 4 -- Train Attacker
python3 train_atk.py
```

Training hyperparameters are configured as constants at the top of each training script:

| Parameter | Defender (`train_def.py`) | Attacker (`train_atk.py`) |
|-----------|--------------------------|--------------------------|
| Algorithm | DQN | TD3 |
| Total timesteps | 1,000,000,000 | 200,000,000 |
| Checkpoint interval | 100,000 | 20,000 |
| Exploration | epsilon 1.0 -> 0.3 | Normal noise (sigma=0.1) |

Model checkpoints are saved to `models/` and TensorBoard logs to `logs/`.

### Monitoring training

```bash
tensorboard --logdir=logs
```

Then open http://localhost:6006 in a browser.

## Configuration

Global settings are in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `HOST` | localhost | Router bind address |
| `PORT` | 5434 | Router bind port |
| `PACKET_SIZE` | 8196 | Max TCP receive buffer (bytes) |
| `INITIAL_RTT` | 0.01 | Initial round-trip time estimate (seconds) |
| `MAX_RETRIES` | 10 | Max send/receive retry attempts |

## Key Results

- The Defender achieves >93% accuracy within ~1 hour of training (~300k steps)
- The Attacker saturates at minimum reward once the Defender learns IP-based filtering, but shows occasional reward spikes suggesting exploration of novel strategies (e.g., at ~2M steps / ~21 hours)
- In all experiments, the Defender consistently outperformed the Attacker

## Troubleshooting

<details>
<summary>Attacker randomly loses connection</summary>
Restart <code>router.py</code>.
</details>

<details>
<summary>Agents fail to connect</summary>
Ensure the router is running and that port 5434 is not in use by another process.
</details>

<details>
<summary>Training appears stuck</summary>
Check TensorBoard for reward progression. Long training runs (40+ hours) are expected for meaningful Attacker learning.
</details>

## Tech Stack

- **Python 3.9** with built-in `socket` and `pickle` for networking
- **OpenAI Gym 0.21** for RL environment interfaces
- **Stable Baselines 3 1.5** for DQN and TD3 implementations
- **TensorFlow 2.8 / PyTorch 1.11** as deep learning backends
- **TensorBoard** for training visualization
- **Black** for code formatting (88 char line width)

## License

Copyright (c) 2022 StAandrew

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
