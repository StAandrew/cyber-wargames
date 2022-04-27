# Cyber War Games
## Setup

#### Repository URL
[GitHub Repository](https://github.com/StAandrew/cyber-wargames)

#### About

This is the main repository for the third-year Project on Network Security with Reinforcement Learning. It is a simulation of Denial-of-Service attack using two separate RL agents implemented with help of OpenAI Gym and Stable Baselines 3 libraries.   
  
Black linter was used with default settings and line width of 88 charactes  

#### Installation

1. Create and activate a venv e.g. `source venv/bin/activate` (Mac) or `.\env\Scripts\activate` (Windows. Note: project was not tested on windows)

2. Install the libraries from requirements.txt e.g. `pip install -r requirements.txt`

## How to use:

To run, run the following in separate terminals and in this order: 
1. `tensorboard --logdir=logs`
2. `python3 router.py`
3. `python3 play_def.py` 
4. `python3 play_atk.py`


## FAQ and Debug

<details>
  <summary>Attacker randomly loses connection </summary>
  Restart the router.py
</details>



#### Copyright(c) 2022 StAandrew

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
