只读复核完成，未修改文件。仍有以下问题，按严重程度排列：

1. **高｜有效样本仅一个时，PPO 会产生 NaN；本次运行已触发。**  
   位置：[campus_ppo.py:207](D:/desk/rl设计/catmouse/campus_ppo.py:207)。  
   `ADV.std()` 默认使用样本标准差，单样本时结果为 NaN，后面的 `+1e-8` 无法修复。真实函数复现中，第一次优化就污染参数，下一轮抛异常。[pool_s2 日志](D:/desk/rl设计/catmouse/results/campus_ppo_pool_s2.log:3) 已出现同样错误，随后热重启；这次崩溃并非驱动重置。

2. **高｜脚本老鼠仍会改写猫的藏身状态，污染学习猫的评估观测。**  
   位置：[campus_scripted.py:111](D:/desk/rl设计/catmouse/campus_scripted.py:111)、[campus_eval.py:68](D:/desk/rl设计/catmouse/campus_eval.py:68)、[campus_ppo.py:127](D:/desk/rl设计/catmouse/campus_ppo.py:127)。  
   `mouse_actions` 的 `need_spot/hide_spot` 更新仍没有老鼠掩码。评估“学习猫 vs 脚本老鼠”时，猫也执行这部分状态更新，藏身点随后进入猫的观测。真实环境最小复现中，10 只猫的藏身点均被改写，最大移动约 **17.36 m**。训练中的 `mouse_vel` 已修好，评估这条路径仍存在同类问题。

3. **高｜子步接触判定与抓捕归属、奖励、统计使用的时刻不一致。**  
   位置：[campus.py:311](D:/desk/rl设计/catmouse/campus.py:311)、[campus_ppo.py:344](D:/desk/rl设计/catmouse/campus_ppo.py:344)、[campus_eval.py:26](D:/desk/rl设计/catmouse/campus_eval.py:26)。  
   已用子步距离判定接触，但 `catcher` 仍按最终距离选择，`near` 和围捕指标也使用最终位置。已复现：
   - 猫 0 先接触，猫 1 后接触，但最终更近，抓捕却记给猫 1。
   - 某猫在首次接触时距受害者 **15.5 m**、最终距 **14 m**，仍获得 `near` 的半份奖励。  
   
   因而个人奖励、混合奖励及“抓捕时”的几何统计仍会错位。

4. **高｜续跑配置失配时，会静默重新训练并覆盖原检查点。**  
   位置：[campus_ppo.py:260](D:/desk/rl设计/catmouse/campus_ppo.py:260)、[campus_ppo.py:403](D:/desk/rl设计/catmouse/campus_ppo.py:403)。  
   沿用同一 tag、更改 `B/share_r/T_ep/pool_max` 等参数时，兼容性判断失败没有报错，随后从第 0 轮训练并覆盖原文件。另外，`T_roll/freeze_mouse/freeze_cat` 未保存和校验，改变这些参数又会被当作正常续跑。

5. **中｜四个子步的端点采样仍可能漏抓擦身接触。**  
   位置：[campus.py:285](D:/desk/rl设计/catmouse/campus.py:285)。  
   默认 `dt=0.5` 下，猫在 `(0,0)`、鼠在 `(0.5,0.4)`，沿 x 方向各以 `4 m/s` 相向移动：连续轨迹最近距离为 **0.4 m**，但采样最近距离约 **0.6403 m**，仍未抓捕。原先的迎面交换位置案例已修复，连续接触检测尚未完整解决。

6. **中｜续跑会清空历史对手使用计数。**  
   位置：[campus_ppo.py:278](D:/desk/rl设计/catmouse/campus_ppo.py:278)。  
   `snaps` 被恢复，已保存的 `use_count` 没有恢复。实际 `pool_s1/s2` 各重启一次，最终计数只覆盖重启之后，不能用于说明全程版本覆盖率和使用频率。

7. **中｜训练重试耗尽后，脚本仍可能报告成功并继续评估。**  
   位置：[run_all2.sh:13](D:/desk/rl设计/catmouse/run_all2.sh:13)、[run_all2.sh:41](D:/desk/rl设计/catmouse/run_all2.sh:41)。  
   第五次失败后，最后的 `sleep` 返回成功，使 `run()` 返回 0；后台任务完成后也未逐项核验训练成功。失效时可能评估未完成的检查点，最后仍记录 `all done`。**本批全部检查点已达到目标轮数，未发现实际发生了重试耗尽。**

其余所述修复——三个函数的指定阵营掩码、四版本分配与 `old_logits` 索引、`pool_max=1`、快照时序、`surv_a/b`、独立评估种子及调用顺序复现性——均未发现错误。已核查全部 **21 份最终检查点**及历史快照，参数均为有限值；上述 NaN 没有留在最终保存的权重中。

结论：现有结果可作探索性参考，但尚不能作为可信的最终定论。