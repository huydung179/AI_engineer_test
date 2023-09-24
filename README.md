# Trimble/Bilberry: AI Engineer technical exercise

## Environment setup

I used PyTorch 2.0.1+cu118 to develop this project. I assume that you have correctly installed at least one Pytorch version on your system-site-packages and also added the CUDA path to your `PATH` environment variable. To install dependencies, I used the buildin `venv` module as follows:

```bash
python -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Wandb setup
I used wandb to log the training process. To use wandb, you need to create an account and login to your account. Then, you need to create a project and get the API key. Finally, you need to run the following command to login to your account:
```bash
wandb login [your_api_key]
```

## Training
To run the sweep to find the best model architecture, hyperparameters tuning, you first need to change the arguments in `config.yaml` to your own. The `--data_dir` is the data directory containing the `fields` and `roads` folders. The `--save_dir` is the directory to save the model checkpoints.

Next, you can run the following commands to run the sweep:
```bash
wandb sweep --project [your_project_name] config.yaml
# Copy the output command of the previous command and run it
# This command is similar to the one below
wandb agent [your_project_name]/[your_sweep_id]
```
By doing this, your sweep will be run and the log will be saved to your wandb project.

## Evaluation
To evaluate the model, you can run the following command:
```bash
python evaluate.py --data_dir [your_data_dir] --model_path [your_model_path] --model_name [your_model_name] --batch_size [your_batch_size]
```
The `--data_dir` is the data directory containing the `fields` and `roads` folders.
The `--model_path` is the path to the model checkpoint you want to evaluate.
The `--model_name` is the name of the model architecture you want to use. It is one of the architecture names in `config.yaml`. This model architecture must match the model architecture of the model checkpoint you want to evaluate.
The `--batch_size` is the batch size you want to use to evaluate the model.