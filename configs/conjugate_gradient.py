seed_everything: 42

model: 
  class_path: "dlboost.tasks.XDGRASP.Recon"
  init_args:
    nufft_im_size: [320, 320]
    patch_size: &PATCH_SIZE [8, 320, 320]
    lambda1: 0.01
    lambda2: 0.01
    lr: 1e-5

data: 
  class_path: "dlboost.datasets.XDGRASP.DCE_XDGRASP_KXKYZ"
  init_args: 
    data_dir: "/data/anlab/Chunxu/RawData_MR/"
    train_scope: [0,17]
    val_scope: [17,18]
    load_scope: [0,-1]
    num_samples_per_subject: 80
    patch_size: *PATCH_SIZE
    train_batch_size: 1
    eval_batch_size: 1
    num_workers: 0

trainer:
  accelerator: "gpu"
  devices: "1"
  # logger: 
  #   class_path: lightning.pytorch.loggers.WandbLogger
  #   init_args:
  #     name: &MODEL_NAME "MOTIF_CORD_CSM_MVF"
  #     project: "MOTIF_CORD"
  callbacks: 
  - class_path: "lightning.pytorch.callbacks.ModelCheckpoint"
    init_args:
      dirpath: &ROOT_DIR "./experiments/XDGRASP_CSM"
      save_top_k: -1
      save_last: true
      every_n_epochs: 1
      save_weights_only: false
      filename: "{epoch:02d}"
  - class_path: "lightning.pytorch.callbacks.RichProgressBar"
  log_every_n_steps: 10
  # fast_dev_run: true
  # overfit_batches: 0.0
  # inference_mode: true
  default_root_dir: *ROOT_DIR
  num_sanity_val_steps: 0
  check_val_every_n_epoch: 100
