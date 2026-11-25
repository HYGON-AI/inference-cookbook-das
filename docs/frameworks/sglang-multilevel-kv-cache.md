# SGLang 多级 KV Cache（L2 / L3 / L4）使用文档

SGLang 目前有六档 KV 缓存的存储配置：

· **L1（仅 GPU）**：KV 缓存完全驻留在 GPU 显存中。

· **L1+L2（L1 + HiCache L2）**：KV 缓存通过 HiCache 的两级架构跨 GPU 和主机内存分布。

· **L1+L2+L3（L1 + HiCache L2 + Mooncake L3）**：KV 缓存进一步扩展至 Mooncake 分布式内存池。

· **L1+L2+L3+L4（L1 + HiCache L2 + Mooncake L3 + Local SSD）**：在 L1+L2+L3 之上，Mooncake 再把冷 KV 从分布式内存池 offload 到本地 SSD。

· **L1+L3（L1 + Mooncake L3）**：KV 缓存跨 GPU 和 Mooncake 分布式内存池分布，不经过主机内存。

· **L1+L3+L4（L1 + Mooncake L3 + DFS）**：KV 缓存跨 GPU 和 Mooncake 分布式内存池以及分布式存储分布，不经过主机内存。

---

|档位|数据路径|Mooncake|SGLang 开关|
|---|---|---|---|
|L1|GPU|否|无|
|L1+L2|GPU → 主机 DRAM|否|--enable-hierarchical-cache|
|L1+L2+L3|GPU → L2 →<br>Mooncake DRAM|要|--enable-hierarchical-cache + <br>--hicache-storage-backend mooncake|
|L1+L2+L3+L4|GPU → L2 →<br>Mooncake DRAM → SSD|要|--enable-hierarchical-cache + <br>--hicache-storage-backend mooncake，<br>master 和 client 中要设置 enable_offload=true|
|L1+L3|GPU → Mooncake DRAM|要|--enable-unified-cache-external-linker + <br>--unified-cache-external-linker-backend mooncake|
|L1+L3+L4|GPU →<br>Mooncake DRAM → DFS|要|--enable-unified-cache-external-linker +<br>--unified-cache-external-linker-backend mooncake，<br>master 和 client 中要设置 enable_offload=true<br>以及 DFS 环境变量|

> 有 Mooncake 的档位：先起 master。L3 容量可以由独立 `mooncake_client` 出，也可以由 SGLang 自己当 client 出。
>
> HiCache 和 L3 直通互斥，不要混开。

- [1. 配置方法](#1-配置方法)
  - [1.1 公共配置](#11-公共配置)
    - [1.1.1 Metadata 选一种](#111-metadata-选一种)
    - [1.1.2 SGLang 自己当 Mooncake client](#112-sglang-自己当-mooncake-client)
  - [1.2 L1+L2（HiCache）](#12-l1l2hicache)
  - [1.3 L1+L2+L3（HiCache + Mooncake）](#13-l1l2l3hicache-mooncake)
  - [1.4 L1+L2+L3+L4（HiCache + Mooncake + Local SSD）](#14-l1l2l3l4hicache-mooncake-local-ssd)
  - [1.5 L1+L3（Mooncake）](#15-l1l3mooncake)
  - [1.6 L1+L3+L4（Mooncake+DFS）](#16-l1l3l4mooncakedfs)
- [2. DeepSeek-V4-Flash 最佳实践](#2-deepseek-v4-flash-最佳实践)
  - [2.1 Mooncake master](#21-mooncake-master)
  - [2.2 Mooncake client](#22-mooncake-client)
  - [2.3 各档共用的模型环境变量](#23-各档共用的模型环境变量)
  - [2.4 L1+L2](#24-l1l2)
  - [2.5 L1+L2+L3](#25-l1l2l3)
  - [2.6 L1+L2+L3+L4 (Local SSD)](#26-l1l2l3l4-local-ssd)
  - [2.7 L1+L3](#27-l1l3)
  - [2.8 L1+L3+L4 (DFS)](#28-l1l3l4-dfs)
  - [2.9 SGLang 自己当 client](#29-sglang-自己当-client)

## 1. 配置方法

### 1.1 公共配置

**这节是 Mooncake master / client 以及 SGLang 公共的参数和设置（含 metadata 四种形式）。后面每档（1.2 — 1.6）只写相对公共部分多出来的配置。**

Master 管元数据和淘汰。DRAM 容量（`global_segment_size`）必须有进程贡献，两种做法：

- **独立 client**：起 `mooncake_client`，SGLang 设 `MOONCAKE_GLOBAL_SEGMENT_SIZE=0`，只发 RDMA 请求。

- **SGLang 自己当 client**：不起 `mooncake_client`，SGLang 设 `MOONCAKE_GLOBAL_SEGMENT_SIZE>0`，见第 1.1.2 节。

`--host` 填本机 IPv4，不能写 `0.0.0.0`：

```Bash
local_host_ip=$(ip -4 route get 8.8.8.8 | awk '{for(i=1;i<=NF;i++) if($i=="src") {print $(i+1); exit}}')
```

#### 1.1.1 Metadata 选一种

下面列举每种 Metadata 形式的 master 以及独立形式的 client。四种形式只改 master 怎么起、独立 client 的两个地址以及 SGLang 的两个地址。

建议：`curl http://127.0.0.1:9300/health` 通了再起 SGLang。

##### P2P Handshake

开发 / 小集群，不需要 metadata 服务。

```Bash
mooncake_master --rpc_port=50051

mooncake_client \
  --host=${local_host_ip} \
  --port=50052 \
  --master_server_address=${master_ip}:50051 \
  --metadata_server=P2PHANDSHAKE \
  --protocol=rdma \
  --device_names="" \
  --global_segment_size=100GB \
  --local_buffer_size=0 \
  --threads=4 \
  --tenant_id=default \
  --enable_http_server=true \
  --http_port=9300

# SGLang 需要设置：
export MOONCAKE_MASTER=${master_ip}:50051
export MOONCAKE_TE_META_DATA_SERVER=P2PHANDSHAKE
export MOONCAKE_LOCAL_HOSTNAME=${local_host_ip}
export MOONCAKE_PROTOCOL=rdma
export MOONCAKE_GLOBAL_SEGMENT_SIZE=0
export MOONCAKE_CHECK_SERVER=1
```

##### HTTP Metadata

中小集群，不想运维 etcd。

```Bash
mooncake_master \
  --rpc_port=50051 \
  --enable_http_metadata_server=true \
  --http_metadata_server_host=0.0.0.0 \
  --http_metadata_server_port=8080 \
  --eviction_high_watermark_ratio=0.95

mooncake_client \
  --host=${local_host_ip} \
  --port=50052 \
  --master_server_address=${master_ip}:50051 \
  --metadata_server=http://${master_ip}:8080/metadata \
  --protocol=rdma \
  --device_names="" \
  --global_segment_size=100GB \
  --local_buffer_size=0 \
  --threads=4 \
  --tenant_id=default \
  --enable_http_server=true \
  --http_port=9300

# SGLang 需要设置：
export MOONCAKE_MASTER=${master_ip}:50051
export MOONCAKE_TE_META_DATA_SERVER=http://${master_ip}:8080/metadata
export MOONCAKE_LOCAL_HOSTNAME=${local_host_ip}
export MOONCAKE_PROTOCOL=rdma
export MOONCAKE_GLOBAL_SEGMENT_SIZE=0
export MOONCAKE_CHECK_SERVER=1
```

##### etcd（不开 HA）

较大集群。etcd 只给 TE / client 用，master 仍直连。

```Bash
etcd \
  --name default \
  --data-dir /tmp/etcd-mooncake \
  --listen-client-urls http://0.0.0.0:2379 \
  --advertise-client-urls http://${etcd_ip}:2379 \
  --listen-peer-urls http://0.0.0.0:2380 \
  --initial-advertise-peer-urls http://${etcd_ip}:2380 \
  --initial-cluster default=http://${etcd_ip}:2380

mooncake_master --rpc_port=50051

mooncake_client \
  --host=${local_host_ip} \
  --port=50052 \
  --master_server_address=${master_ip}:50051 \
  --metadata_server=etcd://${etcd_ip}:2379 \
  --protocol=rdma \
  --device_names="" \
  --global_segment_size=100GB \
  --local_buffer_size=0 \
  --threads=4 \
  --tenant_id=default \
  --enable_http_server=true \
  --http_port=9300

# SGLang 需要设置：
export MOONCAKE_MASTER=${master_ip}:50051
export MOONCAKE_TE_META_DATA_SERVER=etcd://${etcd_ip}:2379
export MOONCAKE_LOCAL_HOSTNAME=${local_host_ip}
export MOONCAKE_PROTOCOL=rdma
export MOONCAKE_GLOBAL_SEGMENT_SIZE=0
export MOONCAKE_CHECK_SERVER=1
```

##### etcd + HA

master 选主，client 用 etcd 发现 master。至少两台 master，每台 `--rpc_address` 填自己的 IP。

```Bash
etcd \
  --name default \
  --data-dir /tmp/etcd-mooncake \
  --listen-client-urls http://0.0.0.0:2379 \
  --advertise-client-urls http://${etcd_ip}:2379 \
  --listen-peer-urls http://0.0.0.0:2380 \
  --initial-advertise-peer-urls http://${etcd_ip}:2380 \
  --initial-cluster default=http://${etcd_ip}:2380

mooncake_master \
  --enable_ha=true \
  --etcd_endpoints=${etcd_ip}:2379 \
  --rpc_address=${本机IP} \
  --rpc_port=50051

mooncake_client \
  --host=${local_host_ip} \
  --port=50052 \
  --master_server_address=etcd://${etcd_ip}:2379 \
  --metadata_server=etcd://${etcd_ip}:2379 \
  --protocol=rdma \
  --device_names="" \
  --global_segment_size=100GB \
  --local_buffer_size=0 \
  --threads=4 \
  --tenant_id=default \
  --enable_http_server=true \
  --http_port=9300

# SGLang 需要设置：
export MOONCAKE_MASTER=etcd://${etcd_ip}:2379
export MOONCAKE_TE_META_DATA_SERVER=etcd://${etcd_ip}:2379
export MOONCAKE_LOCAL_HOSTNAME=${local_host_ip}
export MOONCAKE_PROTOCOL=rdma
export MOONCAKE_GLOBAL_SEGMENT_SIZE=0
export MOONCAKE_CHECK_SERVER=1
```

其中，**独立 client 和 SGLang **上根据实际情况可能需要设置以下变量：

> 这三项**不是必填**。不设时 Mooncake 会自己探测网卡（`--device_names=""` 即可）。只有要用自定义拓扑时才设：`MC_CUSTOM_TOPO_JSON` 指向 JSON，`MC_MS_AUTO_DISC=1` 才会去读这个文件（否则仍走自动探测），`MC_MS_FILTERS` 用来限定网卡。
> 
> 

```Bash
export MC_CUSTOM_TOPO_JSON=/path/to/topology-best.json
export MC_MS_FILTERS=shca_0,shca_1,shca_2,shca_3
export MC_MS_AUTO_DISC=1
```

#### 1.1.2 SGLang 自己当 Mooncake client

不单独起 `mooncake_client`。仍要起 master（第 1.1.1 节），SGLang 进程内嵌 Real Client，既发请求也贡献 DRAM。SGLang 退出后，它贡献的那份 KV 会丢。

把第 1.1.1 节 SGLang 侧的 `MOONCAKE_GLOBAL_SEGMENT_SIZE=0` 改成容量即可。

SGLang 需要设置：

```Bash
export MOONCAKE_MASTER=127.0.0.1:50051
export MOONCAKE_TE_META_DATA_SERVER=http://127.0.0.1:8080/metadata
export MOONCAKE_LOCAL_HOSTNAME=<本机IP>
export MOONCAKE_PROTOCOL=rdma
export MOONCAKE_GLOBAL_SEGMENT_SIZE=100GB
```

---

### 1.2 L1+L2（HiCache）

不起 master/client。在第 1.1.1 节 `sglang serve` 公共部分上加：

```Bash
--enable-hierarchical-cache \
--hicache-ratio 2 \
--hicache-mem-layout page_first_direct \
--hicache-io-backend direct \
--hicache-write-policy write_through
```

---

### 1.3 L1+L2+L3（HiCache + Mooncake）

起 master，需要独立 client 时起 client，在第 1.1.1 节 `sglang serve` 公共部分上加：

```Bash
--enable-hierarchical-cache \
--hicache-ratio 2 \
--hicache-mem-layout page_first_direct \
--hicache-io-backend direct \
--hicache-write-policy write_through \
--hicache-storage-backend mooncake \
--hicache-storage-prefetch-policy timeout
```

---

### 1.4 L1+L2+L3+L4（HiCache + Mooncake + Local SSD）

master 相对第 1.1.1 节加 `--enable_offload=true`：

```Plain Text
mooncake_master \
  --rpc_port=50051 \
  --enable_offload=true
```

**独立 client**

client 相对第 1.1.1 节加 `--enable_offload=true`，以及 MOONCAKE_OFFLOAD 环境变量：

```Bash
export MOONCAKE_OFFLOAD_FILE_STORAGE_PATH="/mnt/data/file_storage"
export MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=21474836480  # SSD offload buffer 20GB
export MOONCAKE_OFFLOAD_USE_URING=1

mooncake_client \
  --host=${local_host_ip} \
  --port=50052 \
  --master_server_address=${master_ip}:50051 \
  --metadata_server=http://${master_ip}:8080/metadata \
  --protocol=rdma \
  --device_names="" \
  --global_segment_size=100GB \
  --local_buffer_size=0 \
  --threads=4 \
  --tenant_id=default \
  --enable_http_server=true \
  --http_port=9300 \
  --enable_offload=true
```

SGLang 和第 1.3 节 L1+L2+L3 一样，在第 1.1.1 节 `sglang serve` 公共部分上加：

```Bash
--enable-hierarchical-cache \
--hicache-ratio 2 \
--hicache-mem-layout page_first_direct \
--hicache-io-backend direct \
--hicache-write-policy write_through \
--hicache-storage-backend mooncake \
--hicache-storage-prefetch-policy timeout
```

**SGLang 自己当 client**

SGLang 内嵌 Real Client 时，需要设置 “MOONCAKE_ENABLE_SSD_OFFLOAD” 和对应其他参数。不起 `mooncake_client`，master 同样 `--enable_offload=true`。在第 1.1.2 节 `sglang serve` 公共部分上加：

```Plain Text
export MOONCAKE_ENABLE_SSD_OFFLOAD=1
export MOONCAKE_OFFLOAD_FILE_STORAGE_PATH="/mnt/data/file_storage"
export MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=21474836480
export MOONCAKE_OFFLOAD_USE_URING=1

--enable-hierarchical-cache \
--hicache-ratio 2 \
--hicache-mem-layout page_first_direct \
--hicache-io-backend direct \
--hicache-write-policy write_through \
--hicache-storage-backend mooncake \
--hicache-storage-prefetch-policy timeout
```

---

### 1.5 L1+L3（Mooncake）

起 master，需要独立 client 时起 client，在原有 `sglang serve` 上加：

```Bash
--enable-unified-cache-external-linker \
--unified-cache-external-linker-backend mooncake
```

---

### 1.6 L1+L3+L4（Mooncake+DFS）

master 相对第 1.1.1 节加 `--enable_offload=true`，以及：

```Plain Text
export MOONCAKE_ENABLE_DFS=1
export MOONCAKE_DFS_FS_ADAPTER=posix
export MOONCAKE_DFS_ROOT_DIR=/stortest/yb_space/dfs_storage/102111010
export MOONCAKE_DFS_ALLOCATOR_TYPE=bucket
export MOONCAKE_DFS_BUCKET_CAPACITY=$((1024 * 1024 * 256))
export MOONCAKE_DFS_MAX_BUCKET_COUNT=5120
export MOONCAKE_DFS_ALIGNMENT=4096
export MOONCAKE_DFS_SINGLE_TENANT=True
export MOONCAKE_DFS_BUCKET_META_LOG_THRESHOLD=$((4 * 1024 * 1024))
export MOONCAKE_DFS_EVICTION_ENABLED=True
export MOONCAKE_DFS_EVICTION_HIGH_WATERMARK=0.80
export MOONCAKE_DFS_EVICTION_LOW_WATERMARK=0.60
export MOONCAKE_DFS_EVICTION_CHECK_INTERVAL=2
export MOONCAKE_DFS_DEFERRED_FREE_SECONDS=10
export MC_STORE_ENABLE_SESSION_CACHE=0
export MC_STORE_ENABLE_DFS_PREFETCH=0

mooncake_master \
  --rpc_port=50051 \
  --eviction_high_watermark_ratio=0.8 \
  --eviction_ratio=0.1 \
  --enable_offload=true
```

HTTP 形式把 `--enable_http_metadata_server` 加到原来的 master 命令上即可。

**独立 client**

client 相对第 1.1.1 节加 `--enable_offload=true`，以及 DFS / offload 环境变量。`MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR` 用 `distributed_storage_backend`（不是本地 SSD）。

```Bash
export MOONCAKE_ENABLE_DFS=1
export MOONCAKE_DFS_FS_ADAPTER=posix
export MOONCAKE_DFS_ROOT_DIR=/stortest/yb_space/dfs_storage/102111010
export MOONCAKE_DFS_ALLOCATOR_TYPE=bucket
export MOONCAKE_DFS_BUCKET_CAPACITY=$((1024 * 1024 * 256))
export MOONCAKE_DFS_MAX_BUCKET_COUNT=5120
export MOONCAKE_DFS_ALIGNMENT=4096
export MOONCAKE_DFS_SINGLE_TENANT=True
export MOONCAKE_OFFLOAD_ENABLED=True
export MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR=distributed_storage_backend
export MOONCAKE_OFFLOAD_FILE_STORAGE_PATH=/stortest/yb_space/dfs_storage/102111010
export MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=21474836480
export MC_STORE_ENABLE_SESSION_CACHE=0
export MC_STORE_ENABLE_DFS_PREFETCH=0

mooncake_client \
  --host=${local_host_ip} \
  --port=50052 \
  --master_server_address=${master_ip}:50051 \
  --metadata_server=http://${master_ip}:8080/metadata \
  --protocol=rdma \
  --device_names="" \
  --global_segment_size=8GB \
  --local_buffer_size=4GB \
  --threads=4 \
  --tenant_id=default \
  --enable_http_server=true \
  --http_port=9300 \
  --enable_offload=true
```

SGLang 在第 1.1.1 节 `sglang serve` 公共部分上加：

```JavaScript
export MOONCAKE_ENABLE_DFS=1
export MOONCAKE_DFS_FS_ADAPTER=posix
export MOONCAKE_DFS_ROOT_DIR=/stortest/yb_space/dfs_storage/102111010
export MOONCAKE_DFS_ALLOCATOR_TYPE=bucket
export MOONCAKE_DFS_BUCKET_CAPACITY=$((1024 * 1024 * 256))
export MOONCAKE_DFS_MAX_BUCKET_COUNT=5120
export MOONCAKE_DFS_ALIGNMENT=4096
export MOONCAKE_DFS_SINGLE_TENANT=True
export MOONCAKE_DFS_BUCKET_META_LOG_THRESHOLD=4194304
export MOONCAKE_DFS_EVICTION_ENABLED=True
export MOONCAKE_ENABLE_SSD_OFFLOAD=1
export MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR=distributed_storage_backend
export MOONCAKE_OFFLOAD_FILE_STORAGE_PATH=/stortest/yb_space/dfs_storage/102111010
export MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=21474836480
export SGLANG_DFS_REPLICA_NUM=1
export SGLANG_MOONCAKE_LOCAL_BUFFER_SIZE=4294967296
export MOONCAKE_DFS_BATCH_READ_THREADS=128
export MC_STORE_ENABLE_SESSION_CACHE=0
export MC_STORE_ENABLE_DFS_PREFETCH=0

--enable-unified-cache-external-linker \
--unified-cache-external-linker-backend mooncake \
--mooncake-enable-page-wise-load \
--mooncake-page-wise-load-threshold 1
```

---

## 2. DeepSeek-V4-Flash 最佳实践

该示例使用 HTTP metadata。

### 2.1 Mooncake master

L1+L2 不需要；L1+L2+L3 / L1+L3 先起这个；L1+L2+L3+L4 和 L1+L3+L4 再加 `--enable_offload=true`，具体的命令见 2.6 和 2.8。

```Bash
mooncake_master \
  --enable_http_metadata_server=true \
  --http_metadata_server_port=8080 \
  --eviction_high_watermark_ratio=0.95
```

### 2.2 Mooncake client

master 起来后再起。

```Bash
WORK=dpsk_v4_flash
local_host_ip=$(ip -4 route get 8.8.8.8 | awk '{for(i=1;i<=NF;i++) if($i=="src") {print $(i+1); exit}}')

export MC_CUSTOM_TOPO_JSON=$WORK/assets/topology-best.json
unset MC_TOPO_FILE_FORCE
export MC_MS_FILTERS=shca_0,shca_1,shca_2,shca_3
export MC_IB_GID_INDEX=0
export MC_MS_AUTO_DISC=1

mooncake_client \
  --host=${local_host_ip} \
  --port=50052 \
  --master_server_address=127.0.0.1:50051 \
  --metadata_server=http://127.0.0.1:8080/metadata \
  --protocol=rdma \
  --device_names="" \
  --global_segment_size=100GB \
  --local_buffer_size=0 \
  --threads=4 \
  --tenant_id=default \
  --enable_http_server=true \
  --http_port=9300
```

> 建议： curl http://127.0.0.1:9300/health 通了再起 SGLang。
> 
> 

### 2.3 各档共用的模型环境变量

serve 脚本里这部分相同，后面不再重复：

```Bash
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export SGLANG_OPT_USE_FUSED_STORE_CACHE=false
export SGLANG_OPT_USE_FUSED_HASH_TOPK=true
export SGLANG_OPT_SWIGLU_CLAMP_FUSION=false
export SGLANG_TOPK_TRANSFORM_512_TORCH=false
export SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=true
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_USE_AITER_AG=0
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_HEAP_SIZE=3173741824
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export SGLANG_ROCM_USE_AITER_MOE=1
export SGLANG_USE_OPT_CAT=1
export SGLANG_USE_FUSED_MLA_CAT=1
export SGLANG_USE_LIGHTOP_GROUP_FP8_QUANT=0
export SGLANG_USE_FUSED_DPSKV4_SILU_MUL_FP8_QUANT=0
export SGLANG_USE_LINEAR_BF16_FP32_USE_BLASLT=1
export SGLANG_APPLY_CONFIG_BACKUP=none
export SGLANG_ROCM_USE_AITER_TILELANG_MHC=1
export SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA=1
export SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=0
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_DPSKV4_LIGHTOP_QUANT_K_CACHE=1
export SGLANG_USE_DPSKV4_LIGHTOP_RMSNORM=1
export SGLANG_USE_FUSED_DPSKV4_QNORM_ROPE_KV_ROPE_QUANT=1
export SGLANG_DSV4_HCU_INT8_INDEX_K_CACHE=1
export SGLANG_LIGHTOP_TOPK=1
export SGLANG_USE_LIGHTOP_EP_MOE_ALIGN=1
export SGLANG_USE_LIGHTOP_EP_SCATTER=1
export SGLANG_USE_LIGHTOP_EP_GATHER=1
export SGLANG_USE_LIGHTOP_TOPK_IDS_POSTPROCESS=1
export SGLANG_OPT_FP8_WO_A_GEMM=0
export SGLANG_RAGGED_VERIFY_MODE=static
export SGLANG_DSPARK_CONFIDENCE_RELAY_LAG_STEPS=2
export SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD=1
export SGLANG_DSPARK_ENABLE_MULTI_STREAM=1
export SGLANG_DSPARK_FAST_KERNEL=1
export SGLANG_DSPARK_FAST_SAMPLING=1
export SGLANG_W4A8_TPMOE_BACKEND=aiter
export SGLANG_UVICORN_WORKER_HEALTHCHECK_TIMEOUT=120
```

### 2.4 L1+L2

不需要起 Mooncake。

```Bash
export SGLANG_KV_LAYOUT_HCU_FA=0

sglang serve \
  --reasoning-parser deepseek-v4 \
  --tool-call-parser deepseekv4 \
  --tp-size 8 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --port 30001 \
  --host 0.0.0.0 \
  --model-path /models/DeepSeek-V4-Flash-0731-W4A8-INT4-Channel-Attn-W8A8-INT8-Channel \
  --model-loader-extra-config '{"enable_multithread_load": "true", "num_threads": 64}' \
  --trust-remote-code \
  --chunked-prefill-size 32768 \
  --disable-flashinfer-autotune \
  --skip-server-warmup \
  --cuda-graph-max-bs 32 \
  --mem-fraction-static 0.9 \
  --speculative-algorithm DSPARK \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --max-running-requests 32 \
  --enable-metrics \
  --enable-cache-report \
  --swa-full-tokens-ratio 0.9 \
  --moe-runner-backend aiter \
  --quantization slimquant_marlin \
  --dp 8 \
  --enable-dp-attention \
  --enable-dp-lm-head \
  --moe-a2a-backend none \
  --tokenizer-worker-num 4 \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-mem-layout page_first_direct \
  --hicache-io-backend direct \
  --hicache-write-policy write_through
```

### 2.5 L1+L2+L3

先起 2.1 / 2.2。相对 L1+L2 多了 Mooncake 环境变量，以及 `--hicache-storage-backend mooncake`。

```Bash
WORK=dpsk_v4_flash
local_host_ip=$(ip -4 route get 8.8.8.8 | awk '{for(i=1;i<=NF;i++) if($i=="src") {print $(i+1); exit}}')

export MOONCAKE_MASTER=127.0.0.1:50051
export MOONCAKE_TE_META_DATA_SERVER=http://127.0.0.1:8080/metadata
export MOONCAKE_LOCAL_HOSTNAME=${local_host_ip}
export MOONCAKE_PROTOCOL=rdma
export MOONCAKE_DEVICE=
export MOONCAKE_GLOBAL_SEGMENT_SIZE=0
export MOONCAKE_TENANT_ID=default
export MOONCAKE_CHECK_SERVER=1
export MC_CUSTOM_TOPO_JSON=$WORK/assets/topology-best.json
unset MC_TOPO_FILE_FORCE
export MC_MS_FILTERS=shca_0,shca_1,shca_2,shca_3
export MC_IB_GID_INDEX=0
export MC_MS_AUTO_DISC=1
export SGLANG_KV_LAYOUT_HCU_FA=0

sglang serve \
  --reasoning-parser deepseek-v4 \
  --tool-call-parser deepseekv4 \
  --tp-size 8 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --port 30001 \
  --host 0.0.0.0 \
  --model-path /models/DeepSeek-V4-Flash-0731-W4A8-INT4-Channel-Attn-W8A8-INT8-Channel \
  --model-loader-extra-config '{"enable_multithread_load": "true", "num_threads": 64}' \
  --trust-remote-code \
  --chunked-prefill-size 32768 \
  --disable-flashinfer-autotune \
  --skip-server-warmup \
  --cuda-graph-max-bs 32 \
  --mem-fraction-static 0.9 \
  --speculative-algorithm DSPARK \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --max-running-requests 32 \
  --enable-metrics \
  --enable-cache-report \
  --swa-full-tokens-ratio 0.9 \
  --moe-runner-backend aiter \
  --quantization slimquant_marlin \
  --dp 8 \
  --enable-dp-attention \
  --enable-dp-lm-head \
  --moe-a2a-backend none \
  --tokenizer-worker-num 4 \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-mem-layout page_first_direct \
  --hicache-io-backend direct \
  --hicache-storage-backend mooncake \
  --hicache-write-policy write_through \
  --hicache-storage-prefetch-policy timeout
```

### 2.6 L1+L2+L3+L4 (Local SSD)

master / 独立 client 加 `--enable_offload=true`。SSD offload buffer 20GB。

```Bash
mooncake_master \
  --enable_http_metadata_server=true \
  --http_metadata_server_port=8080 \
  --eviction_high_watermark_ratio=0.95 \
  --enable_offload=true
```

```C++
WORK=dpsk_v4_flash
local_host_ip=$(ip -4 route get 8.8.8.8 | awk '{for(i=1;i<=NF;i++) if($i=="src") {print $(i+1); exit}}')

export MC_CUSTOM_TOPO_JSON=$WORK/assets/topology-best.json
unset MC_TOPO_FILE_FORCE
export MC_MS_FILTERS=shca_0,shca_1,shca_2,shca_3
export MC_IB_GID_INDEX=0
export MC_MS_AUTO_DISC=1

export MOONCAKE_OFFLOAD_FILE_STORAGE_PATH="/mnt/data/file_storage"
export MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=21474836480
export MOONCAKE_OFFLOAD_USE_URING=1

mooncake_client \
  --host=${local_host_ip} \
  --port=50052 \
  --master_server_address=127.0.0.1:50051 \
  --metadata_server=http://127.0.0.1:8080/metadata \
  --protocol=rdma \
  --device_names="" \
  --global_segment_size=100GB \
  --local_buffer_size=0 \
  --threads=4 \
  --tenant_id=default \
  --enable_http_server=true \
  --http_port=9300 \
  --enable_offload=true
```

```Bash
WORK=dpsk_v4_flash
local_host_ip=$(ip -4 route get 8.8.8.8 | awk '{for(i=1;i<=NF;i++) if($i=="src") {print $(i+1); exit}}')

export MOONCAKE_MASTER=127.0.0.1:50051
export MOONCAKE_TE_META_DATA_SERVER=http://127.0.0.1:8080/metadata
export MOONCAKE_LOCAL_HOSTNAME=${local_host_ip}
export MOONCAKE_PROTOCOL=rdma
export MOONCAKE_DEVICE=
export MOONCAKE_GLOBAL_SEGMENT_SIZE=0
export MOONCAKE_TENANT_ID=default
export MOONCAKE_CHECK_SERVER=1
export MC_CUSTOM_TOPO_JSON=$WORK/assets/topology-best.json
unset MC_TOPO_FILE_FORCE
export MC_MS_FILTERS=shca_0,shca_1,shca_2,shca_3
export MC_IB_GID_INDEX=0
export MC_MS_AUTO_DISC=1
export SGLANG_KV_LAYOUT_HCU_FA=0

sglang serve \
  --reasoning-parser deepseek-v4 \
  --tool-call-parser deepseekv4 \
  --tp-size 8 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --port 30001 \
  --host 0.0.0.0 \
  --model-path /models/DeepSeek-V4-Flash-0731-W4A8-INT4-Channel-Attn-W8A8-INT8-Channel \
  --model-loader-extra-config '{"enable_multithread_load": "true", "num_threads": 64}' \
  --trust-remote-code \
  --chunked-prefill-size 32768 \
  --disable-flashinfer-autotune \
  --skip-server-warmup \
  --cuda-graph-max-bs 32 \
  --mem-fraction-static 0.9 \
  --speculative-algorithm DSPARK \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --max-running-requests 32 \
  --enable-metrics \
  --enable-cache-report \
  --swa-full-tokens-ratio 0.9 \
  --moe-runner-backend aiter \
  --quantization slimquant_marlin \
  --dp 8 \
  --enable-dp-attention \
  --enable-dp-lm-head \
  --moe-a2a-backend none \
  --tokenizer-worker-num 4 \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-mem-layout page_first_direct \
  --hicache-io-backend direct \
  --hicache-storage-backend mooncake \
  --hicache-write-policy write_through \
  --hicache-storage-prefetch-policy timeout
```

### 2.7 L1+L3

先起 2.1 / 2.2。没有`--hicache-*`，只加直通开关`--enable-unified-cache-external-linker`，`--unified-cache-external-linker-backend mooncake`；`SGLANG_KV_LAYOUT_HCU_FA=1`。

```Bash
WORK=dpsk_v4_flash
local_host_ip=$(ip -4 route get 8.8.8.8 | awk '{for(i=1;i<=NF;i++) if($i=="src") {print $(i+1); exit}}')

export MOONCAKE_MASTER=127.0.0.1:50051
export MOONCAKE_TE_META_DATA_SERVER=http://127.0.0.1:8080/metadata
export MOONCAKE_LOCAL_HOSTNAME=${local_host_ip}
export MOONCAKE_PROTOCOL=rdma
export MOONCAKE_DEVICE=
export MOONCAKE_GLOBAL_SEGMENT_SIZE=0
export MOONCAKE_TENANT_ID=default
export MOONCAKE_CHECK_SERVER=1
export MC_CUSTOM_TOPO_JSON=$WORK/assets/topology-best.json
unset MC_TOPO_FILE_FORCE
export MC_MS_FILTERS=shca_0,shca_1,shca_2,shca_3
export MC_IB_GID_INDEX=0
export MC_MS_AUTO_DISC=1
export SGLANG_KV_LAYOUT_HCU_FA=1

sglang serve \
  --reasoning-parser deepseek-v4 \
  --tool-call-parser deepseekv4 \
  --tp-size 8 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --port 30001 \
  --host 0.0.0.0 \
  --model-path /models/DeepSeek-V4-Flash-0731-W4A8-INT4-Channel-Attn-W8A8-INT8-Channel \
  --model-loader-extra-config '{"enable_multithread_load": "true", "num_threads": 64}' \
  --trust-remote-code \
  --chunked-prefill-size 32768 \
  --disable-flashinfer-autotune \
  --skip-server-warmup \
  --cuda-graph-max-bs 32 \
  --mem-fraction-static 0.9 \
  --speculative-algorithm DSPARK \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --max-running-requests 32 \
  --enable-metrics \
  --enable-cache-report \
  --swa-full-tokens-ratio 0.9 \
  --moe-runner-backend aiter \
  --quantization slimquant_marlin \
  --dp 8 \
  --enable-dp-attention \
  --enable-dp-lm-head \
  --moe-a2a-backend none \
  --tokenizer-worker-num 4 \
  --enable-unified-cache-external-linker \
  --unified-cache-external-linker-backend mooncake
```

### 2.8 L1+L3+L4 (DFS)

master / 独立 client 加 `--enable_offload=true` 和 DFS 环境变量。SGLang 加 linker backend / page-wise-load。

```JavaScript
export MOONCAKE_ENABLE_DFS=1
export MOONCAKE_DFS_FS_ADAPTER=posix
export MOONCAKE_DFS_ROOT_DIR=/stortest/yb_space/dfs_storage/102111010
export MOONCAKE_DFS_ALLOCATOR_TYPE=bucket
export MOONCAKE_DFS_BUCKET_CAPACITY=$((1024 * 1024 * 256))
export MOONCAKE_DFS_MAX_BUCKET_COUNT=5120
export MOONCAKE_DFS_ALIGNMENT=4096
export MOONCAKE_DFS_SINGLE_TENANT=True
export MOONCAKE_DFS_BUCKET_META_LOG_THRESHOLD=$((4 * 1024 * 1024))
export MOONCAKE_DFS_EVICTION_ENABLED=True
export MOONCAKE_DFS_EVICTION_HIGH_WATERMARK=0.80
export MOONCAKE_DFS_EVICTION_LOW_WATERMARK=0.60
export MOONCAKE_DFS_EVICTION_CHECK_INTERVAL=2
export MOONCAKE_DFS_DEFERRED_FREE_SECONDS=10
export MC_STORE_ENABLE_SESSION_CACHE=0
export MC_STORE_ENABLE_DFS_PREFETCH=0

mooncake_master \
  --enable_http_metadata_server=true \
  --http_metadata_server_port=8080 \
  --eviction_high_watermark_ratio=0.8 \
  --eviction_ratio=0.1 \
  --enable_offload=true
```

```Bash
WORK=dpsk_v4_flash
local_host_ip=$(ip -4 route get 8.8.8.8 | awk '{for(i=1;i<=NF;i++) if($i=="src") {print $(i+1); exit}}')

export MC_CUSTOM_TOPO_JSON=$WORK/assets/topology-best.json
unset MC_TOPO_FILE_FORCE
export MC_MS_FILTERS=shca_0,shca_1,shca_2,shca_3
export MC_IB_GID_INDEX=0
export MC_MS_AUTO_DISC=1

export MOONCAKE_ENABLE_DFS=1
export MOONCAKE_DFS_FS_ADAPTER=posix
export MOONCAKE_DFS_ROOT_DIR=/stortest/yb_space/dfs_storage/102111010
export MOONCAKE_DFS_ALLOCATOR_TYPE=bucket
export MOONCAKE_DFS_BUCKET_CAPACITY=$((1024 * 1024 * 256))
export MOONCAKE_DFS_MAX_BUCKET_COUNT=5120
export MOONCAKE_DFS_ALIGNMENT=4096
export MOONCAKE_DFS_SINGLE_TENANT=True
export MOONCAKE_OFFLOAD_ENABLED=True
export MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR=distributed_storage_backend
export MOONCAKE_OFFLOAD_FILE_STORAGE_PATH=/stortest/yb_space/dfs_storage/102111010
export MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=21474836480
export MC_STORE_ENABLE_SESSION_CACHE=0
export MC_STORE_ENABLE_DFS_PREFETCH=0

mooncake_client \
  --host=${local_host_ip} \
  --port=50052 \
  --master_server_address=127.0.0.1:50051 \
  --metadata_server=http://127.0.0.1:8080/metadata \
  --protocol=rdma \
  --device_names="" \
  --global_segment_size=8GB \
  --local_buffer_size=4GB \
  --threads=4 \
  --tenant_id=default \
  --enable_http_server=true \
  --http_port=9300 \
  --enable_offload=true
```

```SQL
WORK=dpsk_v4_flash
local_host_ip=$(ip -4 route get 8.8.8.8 | awk '{for(i=1;i<=NF;i++) if($i=="src") {print $(i+1); exit}}')

export MOONCAKE_MASTER=127.0.0.1:50051
export MOONCAKE_TE_META_DATA_SERVER=http://127.0.0.1:8080/metadata
export MOONCAKE_LOCAL_HOSTNAME=${local_host_ip}
export MOONCAKE_PROTOCOL=rdma
export MOONCAKE_DEVICE=
export MOONCAKE_GLOBAL_SEGMENT_SIZE=0
export MOONCAKE_TENANT_ID=default
export MOONCAKE_CHECK_SERVER=1
export MC_CUSTOM_TOPO_JSON=$WORK/assets/topology-best.json
unset MC_TOPO_FILE_FORCE
export MC_MS_FILTERS=shca_0,shca_1,shca_2,shca_3
export MC_IB_GID_INDEX=0
export MC_MS_AUTO_DISC=1
export SGLANG_KV_LAYOUT_HCU_FA=0

export MOONCAKE_ENABLE_DFS=1
export MOONCAKE_DFS_FS_ADAPTER=posix
export MOONCAKE_DFS_ROOT_DIR=/stortest/yb_space/dfs_storage/102111010
export MOONCAKE_DFS_ALLOCATOR_TYPE=bucket
export MOONCAKE_DFS_BUCKET_CAPACITY=$((1024 * 1024 * 256))
export MOONCAKE_DFS_MAX_BUCKET_COUNT=5120
export MOONCAKE_DFS_ALIGNMENT=4096
export MOONCAKE_DFS_SINGLE_TENANT=True
export MOONCAKE_DFS_BUCKET_META_LOG_THRESHOLD=4194304
export MOONCAKE_DFS_EVICTION_ENABLED=True
export MOONCAKE_ENABLE_SSD_OFFLOAD=1
export MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR=distributed_storage_backend
export MOONCAKE_OFFLOAD_FILE_STORAGE_PATH=/stortest/yb_space/dfs_storage/102111010
export MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=21474836480
export SGLANG_DFS_REPLICA_NUM=1
export SGLANG_MOONCAKE_LOCAL_BUFFER_SIZE=4294967296
export MOONCAKE_DFS_BATCH_READ_THREADS=128
export MC_STORE_ENABLE_SESSION_CACHE=0
export MC_STORE_ENABLE_DFS_PREFETCH=0

sglang serve \
  --reasoning-parser deepseek-v4 \
  --tool-call-parser deepseekv4 \
  --tp-size 8 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --port 30001 \
  --host 0.0.0.0 \
  --model-path /models/DeepSeek-V4-Flash-0731-W4A8-INT4-Channel-Attn-W8A8-INT8-Channel \
  --model-loader-extra-config '{"enable_multithread_load": "true", "num_threads": 64}' \
  --trust-remote-code \
  --chunked-prefill-size 32768 \
  --disable-flashinfer-autotune \
  --skip-server-warmup \
  --cuda-graph-max-bs 32 \
  --mem-fraction-static 0.9 \
  --speculative-algorithm DSPARK \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --max-running-requests 32 \
  --enable-metrics \
  --enable-cache-report \
  --swa-full-tokens-ratio 0.9 \
  --moe-runner-backend aiter \
  --quantization slimquant_marlin \
  --dp 8 \
  --enable-dp-attention \
  --enable-dp-lm-head \
  --moe-a2a-backend none \
  --tokenizer-worker-num 4 \
  --enable-unified-cache-external-linker \
  --unified-cache-external-linker-backend mooncake \
  --mooncake-enable-page-wise-load \
  --mooncake-page-wise-load-threshold 1
```

### 2.9 SGLang 自己当 client

只起 2.1 的 master。相对 2.5 / 2.7，把 `MOONCAKE_GLOBAL_SEGMENT_SIZE=0` 改成容量，不要跑 2.2。

L1+L3：

```Bash
WORK=dpsk_v4_flash
local_host_ip=$(ip -4 route get 8.8.8.8 | awk '{for(i=1;i<=NF;i++) if($i=="src") {print $(i+1); exit}}')

export MOONCAKE_MASTER=127.0.0.1:50051
export MOONCAKE_TE_META_DATA_SERVER=http://127.0.0.1:8080/metadata
export MOONCAKE_LOCAL_HOSTNAME=${local_host_ip}
export MOONCAKE_PROTOCOL=rdma
export MOONCAKE_DEVICE=
export MOONCAKE_GLOBAL_SEGMENT_SIZE=100GB
export MOONCAKE_TENANT_ID=default
export MC_CUSTOM_TOPO_JSON=$WORK/assets/topology-best.json
unset MC_TOPO_FILE_FORCE
export MC_MS_FILTERS=shca_0,shca_1,shca_2,shca_3
export MC_IB_GID_INDEX=0
export MC_MS_AUTO_DISC=1
export SGLANG_KV_LAYOUT_HCU_FA=1

sglang serve \
  --reasoning-parser deepseek-v4 \
  --tool-call-parser deepseekv4 \
  --tp-size 8 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --port 30001 \
  --host 0.0.0.0 \
  --model-path /models/DeepSeek-V4-Flash-0731-W4A8-INT4-Channel-Attn-W8A8-INT8-Channel \
  --model-loader-extra-config '{"enable_multithread_load": "true", "num_threads": 64}' \
  --trust-remote-code \
  --chunked-prefill-size 32768 \
  --disable-flashinfer-autotune \
  --skip-server-warmup \
  --cuda-graph-max-bs 32 \
  --mem-fraction-static 0.9 \
  --speculative-algorithm DSPARK \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --max-running-requests 32 \
  --enable-metrics \
  --enable-cache-report \
  --swa-full-tokens-ratio 0.9 \
  --moe-runner-backend aiter \
  --quantization slimquant_marlin \
  --dp 8 \
  --enable-dp-attention \
  --enable-dp-lm-head \
  --moe-a2a-backend none \
  --tokenizer-worker-num 4 \
  --enable-unified-cache-external-linker \
  --unified-cache-external-linker-backend mooncake
```

L1+L2+L3：同样只改 `MOONCAKE_GLOBAL_SEGMENT_SIZE=100GB` 和 `SGLANG_KV_LAYOUT_HCU_FA=0`，`sglang serve` 用 2.5 那组 `--enable-hierarchical-cache` / `--hicache-storage-backend mooncake`。

