![logo](./images/haribros.png)
==================

## Intro
Introducing **SYNTHETIC HARIBROS** Team solution

## Features
### 1. Time-Variant GAN
* Minibatch predictions</br>
* Detect unusual patterns</br>
* Learn complex patterns</br>
* Discriminator teaches the Generator until we find optimums for synthetic data</br>

### 2. Generator and Discriminator Loss Functions

## Getting Started

To recreate our application, the easiest way to get started is to clone the repository:

```bash
# Get the latest snapshot
git clone https://github.com/balintdecsi/synthetic-haribros-watts-up.git myproject

# Change directory
cd myproject

# Install dependencies
python3 -m pip install -r requirements.txt
```

Note that you should have access to GPUs to run this code.

<details>
<summary><h2>About our Solution</h2>

Challange of Energy Networks</br>
    • Energy Transition is on the corner</br>
    • Managing renewables demand smart grids</br>
    • Smart meters for smart grids</br>
    • Roll-out and GDPR concerns</br>
    • Synthetic data paves the way</br>

Model architecture decisions...
</summary>

<h2>GANs</h2>

<h3>Pros</h3>
    • Minibatch predictions</br>
    • Detect unusual patterns</br>
    • Learn complex patterns</br>
<h3>Cons</h3>
    • Data quality dependent</br>
    • Resource hungry</br>
    
<h2>LLMs</h2>

<h3>Pros</h3>
    • Reports and summaries</br>
    • Contextual insights</br>
<h3>Cons</h3>
    • Not great for numbers</br>
    • One token at a time</br>
    • Sticks to first part of data</br>

<h2>Generative Adversarial Networks</h2>
    • Discriminator teaches the Generator until we find optimums for synthetic data</br>
    • We chose Time-Variant GANs</br>
    
<h2>Grid search and findings</h2>

<h3>Ideal number of EPOCHs</h3>
    • cca. 1050</br>
<h3>Target:</h3>
    • Generator loss decreases as Discriminator loss increases</br>
    • Find optimum</br>

<h2>Alternatives for optimization</h2>
    • Different training patterns</br>
    • Alternated Optimizers</br>
    • Feature Engineering</br>

<h2>Discriminator and Generator Loss</h2>
    • After cca. step  875 losses diverge in multiple scenarios(5;6)</br>
    • Early stopping is very difficult due to volatility</br>
    • Semi-Manual optimums should be used</br>

<h2>End result</h2>
    • After cca. step  875 losses diverge in multiple scenarios(5;6)</br> 
    • Early stopping is very difficult due to volatility</br>
    • Manual optimums should be used</br>
</details>

<h2>Acknowledgement</h2>
<a href="https://www.mdpi.com/2079-9292/11/3/355">Synthetic Energy Data Generation Using Time Variant Generative Adversarial Network</a></br>
<a href="https://github.com/stefan-jansen/machine-learning-for-trading/tree/main/21_gans_for_synthetic_time_series">Generative Adversarial Nets for Synthetic Time Series Data</a></br>

<h2>Team Members:</h2>
Balint Decsi</br>
Gabor Schwimmer</br>
Tamas Sueli</br>
Zoltan Takacs</br>
