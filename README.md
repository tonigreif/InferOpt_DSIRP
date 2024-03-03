# Combinatorial Optimization and Machine Learning for Dynamic Inventory Routing
We introduce an approach that combines *Combinatorial Optimization (CO)* and *Machine Learning (ML)* to solve inventory routing problems with stochastic demand and dynamic inventory updates. After each inventory update, our approach reduces replenishment and routing decisions to an optimal solution of a *Capacitated Prize-collecting Traveling Salesman Problem (CPCTSP)* for which well-established algorithms exist. Discovering good prize parametrizations is non-trivial; therefore, we have developed a *ML* approach. 

We evaluate the performance of our pipeline in settings with steady-state and more complex demand patterns. Compared to previous works, the policy generated by our algorithm leads to significant cost savings, achieves lower inference time, and can even leverage contextual information.

## Learning with Combinatorial Optimization Layers
The challenge of the *Dynamic And Stochastic Inventory Routing Problem (DSIRP)* arises due to the combinatorially vast nature of both the state space and the decision space. We extended the concept and learning paradigm described in the paper [Learning with Combinatorial Optimization Layers: a Probabilistic Approach](https://arxiv.org/abs/2207.13513). The *CO-enriched ML* pipeline chains a statistical model with a standard *CO* problem for which well-established algorithms exist. We refer to this solution algorithm as oracle. The oracle must share the same set of feasible solutions and be parameterizable.

### CO-layer
We use a *Capacitated Prize-collecting Traveling Salesman Problem (CPCTSP)* as oracle.

### ML-layer
We use a *Physics-informed Neural Network (PINN)* to capture the DSIRP's piecewise linear cost function. The core of all layers is a generalized linear model offering streamlined training and involving a moderate number of interpretable parameters.

## Computational Experiments
Our computational experiments are designed to facilitate comparisons with other approaches. We focus on simple settings, involving a single vehicle, single depot, single commodity, and an order-up-to policy. By thoroughly analyzing the strengths and limitations of our approach, we offer guidance to both practitioners and researchers operating in this field.

## Getting Started
To get started with our algorithm, ensure you have Julia Version 1.8.5 installed. You can find more information about Julia on its main homepage at julialang.org. Follow the steps below:

0. **Install Julia Version 1.8.5**   
1. **Initialize Environment:**
   
   `julia setup_environment.jl`

3. **Train Pipeline:** Set up and train the pipeline with your preferred settings using:
   
   `julia train_pipeline.jl`

5. **Evaluate Pipeline:** Assess the resulting policy's performance in terms of costs and inference time:

   `julia evaluate_pipeline.jl`

6. (Optional) **Evaluate Benchmark:**

   `julia evaluate_benchmark.jl`
   

### Example
Here's an example workflow to demonstrate how to use our approach:

1. `julia setup_environment.jl`
2. `julia train_pipeline.jl --instance_id normal-10_202212-1314-0055-0af411a5-e0e5-486d-9cbf-13fa3c8ece0a --paradigm dagger --pipeline_epochs 1 --update_epochs 1 --num_scenarios 1`

> :warning: Our setting typically uses more epochs and scenarios, but be aware that this leads to long training times.

3. `julia evaluate_pipeline.jl --solution_path dagger/normal/penalty_200/normal-10_202212-1314-0055-0af411a5-e0e5-486d-9cbf-13fa3c8ece0a/240206_152816618_solutions.json --evaluation_horizon 6`

> :warning: Ensure to update the solution_path parameter according to your specific directory structure.

4. `julia evaluate_benchmark.jl --instance_id normal-10_202212-1314-0055-0af411a5-e0e5-486d-9cbf-13fa3c8ece0a --demand_type test --policy mean --evaluation_horizon 6`


## Remark

We strive to continuously update this repository with improvements and enhancements. Your feedback and contributions are greatly appreciated as we work towards enhancing our solution approach.

## Citation

If you use our repository in your research, please cite the following paper:

> [Combinatorial Optimization and Machine Learning for Dynamic Inventory Routing](https://arxiv.org/abs/) - Toni Greif, Louis Bouvier, Christoph M. Flath, Axel Parmentier, Sonja U. K. Rohmer and Thibaut Vidal (2024)

## Related Work and Packages

We build upon existing research in the field, particularly the [Improved branch-and-cut for the Inventory Routing Problem based on a two-commodity flow formulation](https://doi.org/10.1016/j.ejor.2020.08.047) and [A stochastic inventory routing problem with stock-out](https://doi.org/10.1016/j.trc.2011.06.003).

Check out [InferOpt.jl](https://github.com/axelparmentier/InferOpt.jl), a toolbox for using combinatorial optimization algorithms within machine learning pipelines.
It allows you to create differentiable layers from optimization oracles that do not have meaningful derivatives. Typical examples include mixed integer linear programs or graph algorithms.

Additionally, you may find this [repository](https://github.com/tumBAIS/euro-meets-neurips-2022) valuable. It comprises the code to learn a dispatching and routing policy for the Vehicle Routing Problem with Time Windows using a structured learning enriched combinatorial optimization pipeline.
