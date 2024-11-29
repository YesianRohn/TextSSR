lmdb_path="./AnyWord-lmdb/step1_lmdb"
scheduler_path="./model/stable-diffusion-v2-1/scheduler"  # Path to pretrained model or model identifier from huggingface.co/models.
pretrained_vae="./model/vae_ft/checkpoint-150000/vae"
pretrained_unet="./model/stable-diffusion-v2-1/unet"
revision=None  # Revision of pretrained model identifier from huggingface.co/models.
output_dir="./model/step1"  # The output directory where the model predictions and pretrained_model_name_or_path=None  # Path to pretrained model or model identifier from huggingface.co/models.
seed=3407  # A seed for reproducible training.
resolution=256  # The resolution for input images, all the images in the train/validation dataset will be resized to this resolution.
ttf_size=64
max_len=25
language=None
guidance_scale=0.8  # (No help provided for this parameter)
train_batch_size=4  # Batch size (per device) for the training dataloader.
num_train_epochs=100  # Number of training epochs.
max_train_steps=None  # Total number of training steps to perform. If provided, overrides num_train_epochs.
select_data_lenth=100  # Number of images selected for training.
gradient_accumulation_steps=8  # Number of updates steps to accumulate before performing a backward/update pass.
gradient_checkpointing=False  # Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.
learning_rate=0.0001  # Initial learning rate (after the potential warmup period) to use.
scale_lr=False  # Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.
lr_scheduler="constant"  # The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"].
lr_warmup_steps=500  # Number of steps for the warmup in the lr scheduler.
use_8bit_adam=False  # Whether or not to use 8-bit Adam from bitsandbytes.
allow_tf32=False  # Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices.
use_ema=False  # Whether to use EMA model.
non_ema_revision=None  # Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or remote repository specified with --pretrained_model_name_or_path.
dataloader_num_workers=4  # Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
adam_beta1=0.9  # The beta1 parameter for the Adam optimizer.
adam_beta2=0.999  # The beta2 parameter for the Adam optimizer.
adam_weight_decay=0.01  # Weight decay to use.
adam_epsilon=1e-08  # Epsilon value for the Adam optimizer.
max_grad_norm=1.0  # Max gradient norm.
push_to_hub=False  # Whether or not to push the model to the Hub.
hub_token=None  # The token to use to push to the Model Hub.
hub_model_id=None  # The name of the repository to keep in sync with the local `output_dir`.
logging_dir="logs"   # [TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***.
mixed_precision=None  # Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10 and an Nvidia Ampere GPU. Default to the value of accelerate config of the current system or the flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config.
report_to="tensorboard"  # The integration to report the results and logs to. Supported platforms are "tensorboard" (default), "wandb" and "comet_ml". Use "all" to report to all integrations.
checkpointing_steps=10000  # Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming training using `--resume_from_checkpoint`.
checkpoints_total_limit=None  # Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`. See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state for more docs.
resume_from_checkpoint="latest"  # Whether training should be resumed from a previous checkpoint. Use a path saved by `--checkpointing_steps`, or "latest" to automatically select the last available checkpoint.
enable_xformers_memory_efficient_attention=False  # Whether or not to use xformers.