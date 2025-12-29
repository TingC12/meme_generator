import os
import json
from openai import OpenAI
from dotenv import load_dotenv

LIBRARY_FILE = "lora_library.json"
try:
    with open(LIBRARY_FILE, "r", encoding="utf-8") as f:
        lora_library = json.load(f)
        # 轉字串給 AI 看
        lora_list_str = json.dumps(lora_library, indent=2, ensure_ascii=False)
        print(f"成功載入 LoRA 資料庫: {len(lora_library)} 個模型")
except Exception as e:
    print(f"!! 錯誤: 找不到或無法讀取 {LIBRARY_FILE}: {e}")

# 設定 API
load_dotenv()
client = OpenAI()

# 輸出檔案
DECISION_FILE = "brain_decision.json"

# 系統提示詞 (System Prompt) - 這是大腦的 "判斷邏輯"
system_instruction = f"""
You are an AI Director for an image editing agent.
Your goal is to analyze the user's request and make FOUR decisions:

1. **Detection Prompt**: Identify the specific object or body part in the image that needs to be edited.
   - Output a single, specific descriptive phrase (e.g., "face", "car tire").
   - **CRITICAL RULE**: If the user wants to change the **entire image style**, background, or everything, you MUST output exactly: "WHOLE_IMAGE".

2. **LoRA Selection**:
   - Analyze the user's request style (e.g., cute, anime, realistic, sci-fi).
   - Select the most appropriate "Key Name" from the [Available LoRA List] below.
   - If no specific style is requested or fits, select "None".
   - Only select "Detail Tweaker" if the user explicitly asks for "more details", "better quality", or "HD".

3. **Inpainting Prompt**: Write a detailed English prompt for Stable Diffusion. 
   - It should describe the content of the new image.
   - **CRITICAL RULE**: You MUST include the `trigger` words of the selected LoRA in this prompt.
   - **ANIMAL HEAD RULE (CRITICAL)**: 
     - If the user wants to replace a head with an **animal head** (e.g., "catfish head", "lion head"), you MUST emphasize the animal's natural features and **explicitly FORBID human features**.
     - **Bad Prompt**: "a catfish head replacing the human head" (Result: A human-like head with catfish features and hair)
     - **Good Prompt**: "a realistic catfish head, slimy skin, whiskers, no hair, bald, animal head, replacing the human head"
     - Always add "**no hair, bald, animal texture**" to animal head prompts to avoid "hair" from carrying over.

4. **Reference Material (Critical Decision)**:
   - **When to set "true" (Specific Design)**:
     - The user wants to insert a **specific copyrighted character** or **complex mechanical object** (e.g., "Iron Man Helmet", "Darth Vader Mask", "specific logo").
     - The AI model might not know exactly what this specific item looks like without a reference.
   - **When to set "false" (Generic/Common)**:
     - The user wants to change the **art style** only.
     - The user wants to replace a part with a **common animal or object** (e.g., "catfish head", "dog head", "flower", "sunglasses").
     - Stable Diffusion already knows what a "catfish" looks like, so NO external material is needed.
   - **"material_keyword" Rules**:
     - Do NOT just output a character name (e.g., "Iron Man", "Batman").
     - **YOU MUST combine the character name with the specific part.**
     - BAD: "Iron Man" (This generates a whole body/bust)
     - GOOD: "Iron Man Helmet", "Iron Man Mask", "Godzilla Head", "Robot Arm"
     - This keyword is used to generate a crop-out material, so it must be the specific PART only.

**[Available LoRA List]:**
{lora_list_str}

**Output Format (JSON only):**
{{
    "detect_prompt": "head", 
    "lora_key": "Blindbox", 
    "lora_weight": 0.8,
    "final_prompt": "blindbox, chibi, cute, A cute 3d render of...",
    "needs_material": false,
    "material_keyword": "" 
}}
"""

def main():
    print("\n========== [階段 0] 啟動 AI 大腦 (決策模式 v2) ==========")
    
    user_input = input("請輸入您的指令 (例如: '幫這隻狗戴上墨鏡' OR '把這張圖變成吉卜力風格的動畫人物'): ")
    
    if not user_input:
        user_input = "幫他戴個帽子"

    print("大腦正在思考")
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_input}
            ],
            response_format={ "type": "json_object" }, 
            temperature=0.3,
        )
        
        # 解析 JSON
        content = response.choices[0].message.content
        decision_data = json.loads(content)

        # 如果 MANUAL_LORA_KEY 有指定，就覆蓋大腦的選擇
        manual_lora = os.environ.get("MANUAL_LORA_KEY")
        if manual_lora:
            if manual_lora in lora_library:
                print(f"🔧 偵測到 MANUAL_LORA_KEY={manual_lora}，覆蓋大腦的 LoRA 選擇")
                decision_data["lora_key"] = manual_lora
                # 保留原本的權重或給預設 0.8
                decision_data["lora_weight"] = float(decision_data.get("lora_weight", 0.8))
            else:
                print(f"⚠️ MANUAL_LORA_KEY={manual_lora} 不在 lora_library 中，改用大腦自動判斷")
                
        # 提取 LoRA 資訊
        lora_key = decision_data.get("lora_key", "None")
        lora_weight = decision_data.get("lora_weight", 0.8)
        
        print(f"\n[大腦決策]:")
        print(f"  - 搜尋目標 (Detect): {decision_data['detect_prompt']}")
        print(f"  - LoRA 選擇 (Key): {lora_key} (權重: {lora_weight})")
        print(f"  - 繪圖指令 (Prompt): {decision_data['final_prompt']}")
        print(f"  - 需要素材?: {decision_data['needs_material']}")

        # 將決策存成 JSON 檔案
        with open(DECISION_FILE, "w", encoding="utf-8") as f:
            json.dump(decision_data, f, indent=4, ensure_ascii=False)
        
        print(f"決策已儲存至: {DECISION_FILE}")
            
    except Exception as e:
        print(f"!! 大腦當機: {e}")

if __name__ == "__main__":
    main()