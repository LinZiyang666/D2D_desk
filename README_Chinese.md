# 所有步骤以更新版本为准 !!!
# 训练指南
## 环境配置
3台树莓派
试验系统：Ubuntu 22.04
1. 安装miniconda
```sh
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

source ~/miniconda3/bin/activate
conda init --all
```
2. 创建qwencpu环境并配置
```sh
conda create -n qwencpu python=3.11
conda activate qwencpu

pip install torch==2.7.1+cpu torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
如果实验环境为树莓派：
```sh
pip uninstall -y pyarrow pandas numpy datasets

# Install mutually compatible and PI-friendly versions
pip install "numpy==1.26.4" "pandas==2.1.4" "pyarrow==12.0.1" "datasets==2.14.6"

pip install psutil transformers

sudo apt install cgroup-tools

```
其他实验环境：
```sh
pip install psutil transformers datasets

sudo apt install cgroup-tools
```


# PP实验
### 训练命令
在rank0 (master)上运行, master **必须**最先启动
```sh
torchrun --nproc-per-node=1 --nnodes=3 --node-rank=0 --master-addr=[请替换为master ip] --master-port=29500 three_stages_simple_pp.py --train_steps 5 --batch_size 8 --microbatch_num 4 --sudo_pass [请替换为root密码]
```
在rank1上运行
```sh
torchrun --nproc-per-node=1 --nnodes=3 --node-rank=1 --master-addr=[请替换为master ip] --master-port=29500 three_stages_simple_pp.py --train_steps 5 --batch_size 8 --microbatch_num 4 --sudo_pass [请替换为root密码]
```
在rank2上运行
```sh
torchrun --nproc-per-node=1 --nnodes=3 --node-rank=2 --master-addr=[请替换为master ip] --master-port=29500 three_stages_simple_pp.py --train_steps 5 --batch_size 8 --microbatch_num 4 --sudo_pass [请替换为root密码]
```

### 参数讲解
| 参数                 | 数据类型        | 含义                                                                             | 默认值                                              | 典型用法与注意事项                                                                                                                   |
| ------------------ | ----------- | ------------------------------------------------------------------------------ | ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| `--train_steps`    | `int`       | **训练总步数**（优化器更新次数）。<br>省略或设为 `None` 时，脚本按 DataLoader 的长度跑满一个 epoch。            | `None`                                           | *测试/调试* 时可设小一些，如 `--train_steps 200`；<br>若设得 **大于** DataLoader 长度，会在读完数据后抛 `StopIteration`，需自行处理。                           |
| `--batch_size`     | `int`       | **每个 rank（进程）一次前向/反向计算处理的样本数**。<br>全局 batch = `batch_size × 数据并行 rank 数`。      | 读取环境变量 `BATCH_SIZE`；若未设置则为 `16`                  | 扩大可提高吞吐，但显著增加显存占用；在资源受限时减小它。<br>命令行参数 **优先级高于** 环境变量——例如 `--batch_size 32` 会覆盖 `BATCH_SIZE=16`。                             |
| `--microbatch_num` | `int`       | **将一个大 batch 切分成多少个 micro‑batch 进行流水线 (pipeline) 训练**。<br>影响流水线气泡大小、显存峰值和并行效率。 | 读取环境变量 `MICROBATCH_NUM`；若未设置则为 `4`               | 应满足 `batch_size % microbatch_num == 0`，否则最后一个 micro‑batch 不完整。<br>数值越大，流水线越“细”、显存占用越低，但调度/通信开销增加。                           |
| `--sudo_pass` | `str`       | **本机root密码**。|                | 本参数对此实验没有用，但是必须提供                          |

若出现**OOM**请调节batch_size和microbatch_num

## 数据收集
可以通过变更profile_batch来改变记录的batch  
router的带宽可能限制到10mbps量级才有显著差异  
脚本执行完成后会出现形如timeline_batch0_all.json的数据文件  

# DP实验
### 训练命令
在rank0 (master)上运行, master **必须**最先启动
```sh
torchrun --nproc-per-node=1 --nnodes=3 --node-rank=0 --master-addr=[请替换为master ip] --master-port=29500 three_rank_simple_dp.py --train_steps 20 --batch_size 3 --sudo_pass [root_password] --middle_layers [中间层数量]
```
在rank1上运行
```sh
torchrun --nproc-per-node=1 --nnodes=3 --node-rank=1 --master-addr=[请替换为master ip] --master-port=29500 three_rank_simple_dp.py --train_steps 20 --batch_size 3 --sudo_pass [root_password] --middle_layers [中间层数量]
```
在rank2上运行
```sh
torchrun --nproc-per-node=1 --nnodes=3 --node-rank=2 --master-addr=[请替换为master ip] --master-port=29500 three_rank_simple_dp.py --train_steps 20 --batch_size 3 --sudo_pass [root_password] --middle_layers [中间层数量]
```

### 参数讲解
| 参数                 | 数据类型        | 含义                                                                             | 默认值                                              | 典型用法与注意事项                                                                                                                   |
| ------------------ | ----------- | ------------------------------------------------------------------------------ | ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| `--train_steps`    | `int`       | **训练总步数**（优化器更新次数）。<br>省略或设为 `None` 时，脚本按 DataLoader 的长度跑满一个 epoch。            | `None`                                           | *测试/调试* 时可设小一些，如 `--train_steps 200`；<br>若设得 **大于** DataLoader 长度，会在读完数据后抛 `StopIteration`，需自行处理。                           |
| `--batch_size`     | `int`       | **每个 rank（进程）一次前向/反向计算处理的样本数**。<br>全局 batch = `batch_size × 数据并行 rank 数`。      | 读取环境变量 `BATCH_SIZE`；若未设置则为 `16`                  | 扩大可提高吞吐，但显著增加显存占用；在资源受限时减小它。<br>命令行参数 **优先级高于** 环境变量——例如 `--batch_size 32` 会覆盖 `BATCH_SIZE=16`。 在DP实验中必须为3的倍数                            |
| `--middle_layers` | `int`       | **模型transformer部分的层数量** <br> | 需要手动设置               | 树莓派内存无法装下完整的模型，只能装下舍去部分transformer的阉割版模型                        |
| `--sudo_pass` | `str`       | **本机root密码**。|                | 本参数对此实验没有用，但是必须提供                          |

若出现**OOM**请调节batch_size和middle_layers
实验参数应尽最大可能性充分利用内存

## 数据收集
可以通过变更profile_batch来改变记录的batch  
router的带宽可能限制到10mbps量级才有显著差异  
脚本执行完成后会出现形如timeline_batch0_all.json的数据文件  

