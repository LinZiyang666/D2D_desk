# 以下步骤会运行性能测试，如果只需要估计可直接运行asteroid_qwen06b_plan.py

# 步骤 1：在每台机器上测逐层时延与字节

在 8 台机器上分别运行（用本地 Qwen-0.6B 路径；`--host` 用来给机器命名）：

```bash
python measure_layers.py \
  --model_path /path/to/qwen-0_6b \
  --seq_len 2048 \
  --batch 16 \
  --dtype float32 \
  --iters 6 --warmup 2 \
  --out layers_cpu1.json \
  --host cpu1
```

* 输出 `layers_cpu*.json` 包含：

  * `layers[i].param_bytes`：第 i 个 block 的参数字节（**实测自参数个数 × dtype 字节**）
  * `layers[i].activation_bytes_per_sample`：该 block 前后边界激活在每样本的字节数（**实测自 forward 输出张量**）
  * `layers[i].forward_time_s`、`backward_time_s`：该 block 的 fwd/bwd 平均耗时（CPU timer）
  * `embed_param_bytes`、`tail_param_bytes`：词嵌入与尾部（norm+lm\_head）参数字节（**实测**）

> 说明
>
> * backward hook 在部分版本上可能抓不到模块级耗时，脚本会保留 0（下一步会用前向的比例作兜底，或者你也可以切到 torch.profiler 记录更精确的 bwd）。
> * `activation_bytes_per_sample` 是**真实输出大小**，因此后续不会再用公式“估”。

---

# 步骤 2：合并 8 台的测量得到 capacity

在任一机器上把 8 份 JSON 合并：

```bash
python aggregate_capacities.py \
  --inputs layers_cpu1.json layers_cpu2.json ... layers_cpu8.json \
  --ref_host cpu1 \
  --out capacities.json
```

* 输出 `capacities.json` 里是 `capacity[host]`，定义为：**参考机**每层前向时间 / 本机每层前向时间 的中位数。
* 之后延迟函数会按 `tf_ref/capacity[host]`、`tb_ref/capacity[host]` 进行**纯测量比例缩放**（无估值公式）。
* `aggregate_capacities.py` 里算出来的 `capacity[host]` 不是论文里的 $v_d$。它是一个**跨机器“速度缩放因子”**，用来把“参考机器的逐层时延曲线”映射到其他机器上，好让 `latency_of_layer(dev, layer, B)` 能在**不重复逐层建模**的前提下给出每台机器的层时延。

---

# 步骤 3：测链路带宽并生成带宽矩阵

用 iperf3 两两测量（双向最好都测），收集成 CSV（无表头）：

```
src_host,dst_host,gbps
cpu1,cpu2,9.8
cpu2,cpu1,9.7
...
```

转成 JSON：

```bash
python parse_bandwidth_csv.py --csv links.csv --out bandwidth_matrix.json
```

* 输出里是 `(src|dst) -> bytes_per_sec` 的字典。planner 将用**你的实测**矩阵计算组内 AllReduce 与组间激活通信，无需假设“统一带宽”。

---

# 步骤 4：用抽样 span 的真实峰值 RSS 拟合内存模型

内存模型形式（对每个 stage 的 span）：

```
peak_bytes ≈ base + alpha * Σ(weights_bytes) + Kp * Σ(activation_bytes_per_sample) * B
```

我们随机抽若干个 \[s,e) span，实际执行 forward+backward，记录进程 RSS 的增量，然后**最小二乘**拟合出 `base, alpha, Kp`：

```bash
python calibrate_memory_model.py \
  --layers_json layers_cpu1.json \
  --model_path /path/to/qwen-0_6b \
  --samples 20 \
  --out memory_model.json
```

* 其中 `Σ(weights_bytes)`、`Σ(activation_bytes_per_sample)` 都来自**步骤 1 的实测**；`B` 是你真正在测的 micro-batch。
* 这样得到的 `base/alpha/Kp` 完全来自你的机器/模型/配置，不再拍脑袋。

---

# 步骤 5：把所有实测数据喂给 planner 出最终 plan

准备一个设备内存预算文件（GB）：

```json
// devices.json
{"cpu1": 24, "cpu2": 24, "cpu3": 24, "cpu4": 24, "cpu5": 24, "cpu6": 24, "cpu7": 24, "cpu8": 24}
```

然后运行：

```bash
python plan_from_measured.py \
  --layers layers_cpu1.json \
  --capacities capacities.json \
  --bandwidth bandwidth_matrix.json \
  --memory_model memory_model.json \
  --devices devices.json \
  --tied_embed true \
  --max_stages None \
  --out_dir out_measured
```

输出：

* `out_measured/stages.csv`、`steps.csv`、`plan_meta.json`、`summary.txt`
* 这些就是用**纯实测**数据驱动的规划结果（Algorithm 1/2 不变）。

