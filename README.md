# Max-Min Off-Policy Actor-Critic Method Focusing on Worst-Case Robustness to Model Misspecification

This repository is the official implementation of Max-Min Off-Policy Actor-Critic Method Focusing on Worst-Case Robustness to Model Misspecification.

# Requirement

To install requirements:

```setup
pip install -r requirements.txt
```

You will also need to install mujoco, if necessary.
We used mjpro150. These can be installed for free.

# Training

```train
python main.py algorithm=m2td3 environment=HalfCheetahv2-1_4 experiment_name=m2td3_HalfCheetahv2-1_4 algorithm.max_steps=2000000 evaluation.evaluate_interval=100000
```

# Evaluation

```eval
python evaluate_mujoco.py --dir experiments/m2td3_HalfCheetahv2-1_4 --interval 100000 --max_iteration 2000000 --dim_evaluate_point 1
```

The algorithm can be selected from `m2td3`, `soft_m2td3`

With the `environment` option, you can train in various scenarios of mujoco tasks.
Check `configs/environment` to see what scenarios are available.
