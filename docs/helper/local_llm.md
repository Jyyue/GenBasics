# 如何本地部署大模型

参考blog：https://www.cnblogs.com/shook/p/18700561

离线： https://yelog.org/2024/10/10/install-ollama-offline/

https://cuterwrite.top/p/ollama/

http://juejin.cn/post/7347667306460577843

## 选择合适大小的模型

一般来说，模型的大小（参数量）以B为单位，B代表billion。FP16精度的模型推理需要的显存为2倍B，例如DeepSeek-R1-7B需要14G显存。

部署模型需要考虑的因素：内存，带宽，核数，并行支持

> 关于如何在CPU/GPU上查看以上内容
在linux上查看cpu内存 free -h
              total        used        free      shared  buff/cache   available
Mem:            30G        4.6G        1.2G         32M         24G         25G
Swap:           62G        1.9G         60G
*查看内存带宽* dmidecode -t memory
‘# dmidecode 3.1’
/sys/firmware/dmi/tables/smbios_entry_point: Permission denied
Scanning /dev/mem for entry point.
/dev/mem: Permission denied
*查看cpu核心数* lscpu
Architecture:          x86_64
CPU op-mode(s):        32-bit, 64-bit
Byte Order:            Little Endian
CPU(s):                24
On-line CPU(s) list:   0-23
Thread(s) per core:    1
Core(s) per socket:    12
Socket(s):             2
NUMA node(s):          2
Vendor ID:             GenuineIntel
CPU family:            6
Model:                 85
Model name:            Intel(R) Xeon(R) Silver 4214 CPU @ 2.20GHz
Stepping:              7
CPU MHz:               2200.000
BogoMIPS:              4400.00
Virtualization:        VT-x
L1d cache:             32K
L1i cache:             32K
L2 cache:              1024K
L3 cache:              16896K
NUMA node0 CPU(s):     0,2,4,6,8,10,12,14,16,18,20,22
NUMA node1 CPU(s):     1,3,5,7,9,11,13,15,17,19,21,23
Flags:                 fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc aperfmperf eagerfpu pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid dca sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch epb cat_l3 cdp_l3 intel_ppin intel_pt ssbd mba ibrs ibpb stibp ibrs_enhanced tpr_shadow vnmi flexpriority ept vpid fsgsbase tsc_adjust bmi1 hle avx2 smep bmi2 erms invpcid rtm cqm mpx rdt_a avx512f avx512dq rdseed adx smap clflushopt clwb avx512cd avx512bw avx512vl xsaveopt xsavec xgetbv1 cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local dtherm ida arat pln pts pku ospke avx512_vnni spec_ctrl intel_stibp flush_l1d arch_capabilities
看GPU：
*GPU显存* nvidia-smi（NVIDIA GPU）或radeontop（AMD GPU）
*GPU带宽，核数，并行* nvidia-smi -q



## 基于原则提供的一般硬件和deepseek模型大小适配

https://cloud.tencent.com/developer/article/2493853
https://zhuanlan.zhihu.com/p/20803691410
https://zhuanlan.zhihu.com/p/22531033367
https://36kr.com/p/3162062358637056

实际上，显存决定了GPU可以放的模型大小，当显存不够的时候可以调内存，这将导致模型运行变慢，但并非不可运行。

## 理解主流大模型不同版本的能力区别

### deekseek模型迭代史
https://hub.baai.ac.cn/view/43089

https://huggingface.co/deepseek-ai

https://api-docs.deepseek.com/zh-cn/updates


可以看到deepseek的模型迭代：面向应用主要为v系列和r系列（含api接口），以及最新推出的多模态Janus系列

问：这两个系列是否有从更老的code/math/chat（MOE）模型迭代？

在满血v3基础上，用Qwen作为基座蒸馏得到了更小版本。

目前ollama支持deepseek-R1不同大小版本的部署。

### 大模型迭代史

https://agijuejin.feishu.cn/wiki/EG8tw8TTCi9AEqkDdSicOnWtnVf

## 使用ollama部署

ollama：https://ollama.com/library/deepseek-r1:1.5b

遇到问题：
Error: pull model manifest: Get "https://registry.ollama.ai/v2/library/deepseek-r1/manifests/1.5b": dial tcp 172.67.182.229:443: i/o timeout

解决办法：尝试ping服务器，可以访问，随后更新重启ollama后解决问题，参考了https://github.com/ollama/ollama/issues/1859

> 关于ping: 终端A执行 ping B域名 命令，生成一个 ICMP 回显请求包，并将其发送到目标主机B。该包包含源 IP 地址，目标 IP 地址：B 的 IP 地址，ICMP，序列号，数据：通常包含一个时间戳或一些数据，TTL：初始值（例如，64）。B接受当 B 收到回显请求包时，B 会返回一个 ICMP 回显应答包。A 接收这个应答包，并用它来评估网络延迟和连接质量。
回显应答包通常包含以下内容：源 IP 地址：B 的 IP 地址。目标 IP 地址：A 的 IP 地址。ICMP 类型：回显应答（类型 0）。序列号：与发送时相同。A 可以通过此序列号匹配每个请求和应答。数据：应答包将返回 A 发送的数据内容（比如时间戳、数据等）。TTL：B 返回的 TTL 值（该值在 B 收到请求时已经递减）。时间戳：记录发送请求时的时间，帮助计算往返时间（RTT）。
随后A 处理的内容：计算 RTT：通过比较请求发送时的时间和收到回显应答时的时间，A 计算出往返时间（RTT）。判断丢包：如果某个数据包没有得到应答，A 会记录丢包。统计：A 会统计所有数据包的传输情况，计算 丢包率 和 RTT 最小/最大/平均，并输出结果。通过这个过程，A可以判断B的

在线部署：curl下载ollama，ollama pull 模型，然后在计算节点上ollama serve，打开第二个shell登入节点，ollama -v确认ollama运行，ollama run 模型名启动模型
离线部署：https://github.com/ollama/ollama/releases 下载合适的模型，tar -xzvf xxx.tar 解压，直接找到对应位置的ollama运行 ollama serve
查看已经部署的模型：ollama list

## 部署后测试
测试CPU和GPU利用率，token生成时间


## 其他配置

前端：https://sspai.com/post/85193

https://cuterwrite.top/p/integrate-open-webui-ollama-qwen25-local-rag/

采用OpenWebUI配置前端chatbot

首先安装docker

然后拉取镜像，采用默认docker数据卷路径（虚拟机内部，非本地）
docker run -d -p 3000:8080 --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main
其中容器的端口8080倍映射到本地3000，可以通过local host访问

成功运行容器后，启动ollama，然后本地访问http://localhost:3000，可以看到OpenWebUI界面，登录后可以看到本地已经下载好的模型

Plus： openWebUI连接其他以部署模型端口：1. 云服务deepseek api 2. 其他服务器的ollama模型端口




