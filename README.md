
# All steps are subject to the updated version !!!
# Training Guide

## Environment Setup

* **3 Raspberry Pi devices**
* **Tested System**: Ubuntu 22.04

### 1. Install Miniconda

```sh
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

source ~/miniconda3/bin/activate
conda init --all
```

### 2. Create and Configure the `qwencpu` Environment

```sh
conda create -n qwencpu python=3.11
conda activate qwencpu

pip install torch==2.7.1+cpu torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

pip install psutil transformers datasets
```

---

## Running the Experiment

### Training Command

Run on **rank 0 (master)** — the master **must** be started first:

```sh
torchrun --nproc-per-node=1 --nnodes=3 --node-rank=0 --master-addr=[replace with master IP] --master-port=29500 three_stages_simple_pp.py --train_steps 5 --batch_size 8 --microbatch_num 4 --sudo_pass [replace with root password]
```

Run on **rank 1**:

```sh
torchrun --nproc-per-node=1 --nnodes=3 --node-rank=1 --master-addr=[replace with master IP] --master-port=29500 three_stages_simple_pp.py --train_steps 5 --batch_size 8 --microbatch_num 4 --sudo_pass [replace with root password]
```

Run on **rank 2**:

```sh
torchrun --nproc-per-node=1 --nnodes=3 --node-rank=2 --master-addr=[replace with master IP] --master-port=29500 three_stages_simple_pp.py --train_steps 5 --batch_size 8 --microbatch_num 4 --sudo_pass [replace with root password]
```

---

### Parameter Explanation

| Argument           | Type  | Description                                                                                                                                    | Default Value                          | Typical Usage & Notes                                                                                                                                                    |
| ------------------ | ----- | ---------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--train_steps`    | `int` | **Total training steps** (number of optimizer updates). <br>If omitted or set to `None`, the script runs a full epoch based on the DataLoader. | `None`                                 | Set a small value (e.g., `--train_steps 200`) for testing/debugging. <br>If larger than the dataset length, `StopIteration` will be raised and must be handled manually. |
| `--batch_size`     | `int` | **Number of samples each rank (process) processes per forward/backward pass**. <br>Global batch = `batch_size × number of DP ranks`.           | Read from `BATCH_SIZE` env var or `16` | Increasing it improves throughput but increases memory usage. <br>Command-line value overrides env var.                                                                  |
| `--microbatch_num` | `int` | **How many micro-batches to split each batch into for pipeline training**. <br>Affects pipeline bubble size, memory usage, and efficiency.     | Read from `MICROBATCH_NUM` or `4`      | Must satisfy `batch_size % microbatch_num == 0`, or the last micro-batch will be incomplete. <br>Larger values reduce memory but increase scheduling/communication cost. |
| `--sudo_pass`      | `str` | **Root password of the local machine**.                                                                                                        | N/A                                    | This parameter is not used in the experiment but must be provided for compatibility.                                                                                     |

> If you encounter **OOM (Out of Memory)**, try reducing `batch_size` or `microbatch_num`.

---

## Data Collection

You can change `profile_batch` to specify which batch to record.
Bandwidth throttling on the router (e.g., down to 10 Mbps) may be necessary to observe significant performance differences.
After the script finishes, data files like `timeline_batch0_all.json` will be generated.
