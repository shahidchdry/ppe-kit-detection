# 🛡️ PPE Kit Detection - Model Training (Google Colab)

This directory contains everything required to train a state-of-the-art **YOLOv8** model for Personal Protective Equipment (PPE) detection and safety compliance alerting.

---

## 🚀 How to Train on Google Colab (Free GPU)

1. Open [Google Colab](https://colab.research.google.com/).
2. Click **File** -> **Upload notebook** and choose:
   `PPE_Kit_Detection_Training_Colab.ipynb`
3. Enable GPU:
   - Click **Runtime** -> **Change runtime type** -> Choose **T4 GPU** -> **Save**.
4. Run all cells (`Runtime` -> `Run all` or press `Ctrl + F9`).
5. Once training completes:
   - The notebook automatically evaluates mAP metrics and confusion matrices.
   - It will trigger a browser download for `best_ppe_yolov8.pt`.
6. Copy the downloaded `best_ppe_yolov8.pt` into `ppe-kit-detection/model/best.pt` or upload it directly through the website UI!

---

## 🎯 Target Detected Classes

| Class ID | Class Name | Category | Compliance Role |
|---|---|---|---|
| 0 | `Hardhat` | PPE Item | 🟢 Positive Compliance |
| 1 | `Mask` | PPE Item | 🟢 Positive Compliance |
| 2 | `NO-Hardhat` | Violation | 🚨 Direct Safety Violation |
| 3 | `NO-Mask` | Violation | 🚨 Direct Safety Violation |
| 4 | `NO-Safety Vest`| Violation | 🚨 Direct Safety Violation |
| 5 | `Person` | Subject | Required to wear PPE |
| 6 | `Safety Cone` | Site Object| Informational |
| 7 | `Safety Vest` | PPE Item | 🟢 Positive Compliance |
| 8 | `Machinery` | Equipment | Site Safety |
| 9 | `Vehicle` | Vehicle | Site Safety |

---

## 💻 Optional: Local or Remote GPU Training

If you ever wish to train from a terminal on an NVIDIA GPU machine:

```bash
pip install ultralytics roboflow opencv-python pyyaml
python train.py --epochs 50 --batch 16 --model yolov8s.pt --device 0
```
