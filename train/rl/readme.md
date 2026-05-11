现在需要完成 RL 部分的内容

参考本地D:\code\Workplace\bachelor_thesis\thesis\body\03_method.tex的内容，我其实是在完成这个毕设的代码

数据构造和SFT已经完成，现在只差RL，需要你参考论文的描述实现代码。
可以不完全相同，做一些你认为的修改或者优化，或者有些表述不可行，你做一些适当调整，自行决定
大体逻辑要实现，确保能够成功训练模型就好

verl 在/data/250010072/zlh/king/verl，有对应的conda环境，你可以适当调整下载需要的其他依赖
代码和数据放到code_tailored_dataset/train/rl下面

对了，因为我使用的是DAPO算法，它有一个动态采样的过程，
就是在训练模型之前，要先让模型对训练集采样十次，剔除掉全部正确的数据（错误可以保留？因为模型能力进化），用余下的数据进行训练
避免资源浪费

代码执行的逻辑参考：code_tailored_dataset/code/select_reasoning.py
我在 kodecode_exec 和 run_code 实现的，原始的 kodecode会判断 是test_code（pytest）还是 test_in, test_out这种测试样例
test_code只适用于python代码，测试样例是通用的，不过我在生成multi_task的时候已经考虑到了这种情况，其它语言的数据都是测试样例的形式
run_code 支持不同语言代码的执行
你再筛选数据和设计奖励函数的时候需要用到这里的函数

大致需要实现两个脚本：
- 筛选数据，输入是完整数据集，输出（两个对应的）筛选后数据集。 模型部署可以考虑vllm多端部署（4个端口部署4个模型，多线程32*4并行） 生成 --> 执行
- 训练模型 用各自的数据集 DAPO 训练模型


现在SFT还在进行中，判断下大致还有多少时间
先配置好环境，写好代码，然后等待SFT完成后，直接进行RL