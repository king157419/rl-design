你是《博弈论与强化学习》课程报告的审稿人。请从多个维度严格审核这份课程设计报告，并给出可执行的修改意见（中文回答）。

材料（只读，不要修改任何文件）：
- 报告 LaTeX 源文件：report/main.tex（编译结果 report/main.pdf，12 页）
- 简略版：report/简略版.tex
- 实验代码：catmouse/ 目录，重点是 campus.py（环境与发现模型）、campus_scripted.py（脚本策略）、campus_ppo.py（双方共同训练的 PPO）、campus_eval.py（评估与围捕指标）、campus_probe.py（脚本实验）、stats_campus.py、stats_coevo.py
- 实验结果：catmouse/results/campus_stats.txt、catmouse/results/campus_coevo_stats.txt、catmouse/results/campus_eval.log、catmouse/results/campus_probe.log

请按下面维度逐条审核，每条写明：问题、证据（文件:行号或报告中的原句）、严重程度（高/中/低）、具体改法。
1. 博弈建模是否正确：POSG 定义、一般和博弈的说法、负外部性/社会困境的论证是否站得住。
2. 强化学习实现是否正确：PPO 损失、GAE 中角色掩码 m_{t+1} 的处理、老鼠被抓后转由猫网络控制时的回报与终止处理、团队奖励的实现、观测是否泄露不该看到的信息。
3. 实验设计与统计：配对比较、自举区间、样本量、对照组（例如“把系数改成 1”的反事实设计是否讲清楚了目的）、交叉评估是否公平、围捕指标是否受猫数量混淆。
4. 结论是否被数据支持：有没有过度解读（例如“团队奖励让猫变弱”“信用分配”“懒惰智能体”的解释），有没有遗漏的替代解释。
5. 报告写作：逻辑衔接、术语是否对课程读者友好、相关工作和本文的关联是否清楚、图表与正文是否一致、数字是否前后一致（请核对正文数字与 results/*.txt）。
6. 代码与报告不一致的地方（参数、规则、指标定义）。

最后给出“必须改”的前 10 条清单，按重要性排序。
