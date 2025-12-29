import os
import time
import json

program_start = time.time()
print("========== [第一棒] AI 大腦：決策中 ==========")
os.system("python run_stage0_llm.py") 

# 讀取決策
need_material = False
if os.path.exists("brain_decision.json"):
    with open("brain_decision.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        need_material = data.get("needs_material", False)

# 判斷是否要插隊
if need_material:
    print("\n========== [插隊 A] 生成原始素材 ==========")
    # 這一步會產生 reference_material.png (有白底)
    os.system("python run_stage1_5_material.py")
    
    print("\n========== [插隊 B] 精修素材 (Florence+SAM) ==========")
    # [關鍵] 呼叫偵測程式，但加上 'material' 參數，讓它去切素材
    os.system("python run_florence_plus_sam.py material")
    
    time.sleep(1)
else:
    print("\n========== [跳過] 不需要素材 ==========")

print("\n========== [第二棒] 偵測主圖目標 ==========")
# 不加參數，預設就是切主圖 (test.webp)
os.system("python run_florence_plus_sam.py")
time.sleep(1)

print("\n========== [第三棒] 繪圖與合成 ==========")
os.system("python run_stage3_inpaint_test1.py")

# 時間計算
total_seconds = time.time() - program_start
print(f"\n總執行時間: {int(total_seconds // 60)} 分 {total_seconds % 60:.2f} 秒")