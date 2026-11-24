"""修复脚本老鼠改写猫藏身状态后，只重评“AI 猫 vs 脚本老鼠”这一项，写回各自的评估文件。"""
import sys
import torch
import campus_eval as E

for tag in sys.argv[1:]:
    path = f"results/eval/{tag}.pt"
    out = torch.load(path, weights_only=False)
    ck = torch.load(f"results/campus_ppo/{tag}.pt")
    k = f"{tag}/learned_cat_vs_mixed_mouse"
    out[k] = E.match("learned", "mixed", ck=ck)
    E.summarize(k, out[k])
    out["_lm_refixed"] = True
    torch.save(out, path)
