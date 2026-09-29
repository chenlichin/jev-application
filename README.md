# Kaggle × Jev：用 Jev 模型重做 5 個經典 ML 題目

把 5 個經典 Kaggle 題目的傳統解法，改成用 [TypeSafe](https://typesafe.ai) 的 **Jev**（System One 模型）做**零樣本推論**，
在同一份測試集上與 Kaggle 經典 baseline 比較效果。

| # | Kaggle 題目 | 類型 | Kaggle 經典解法（baseline） | Jev 問法 |
| --- | --- | --- | --- | --- |
| 1 | [Titanic](https://www.kaggle.com/competitions/titanic) | 表格・二分類 | HistGradientBoosting + Title/FamilySize/HasCabin 特徵工程 | `Noul`：這位乘客是否生還？ |
| 2 | [SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) | 文字・二分類 | TF-IDF + Multinomial Naive Bayes | `Noul`：這則簡訊是否為垃圾訊息？ |
| 3 | [IMDB 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) | 文字・情感分析 | TF-IDF (1-2 gram) + Logistic Regression | `Noul`：整體是否為正面評論？ |
| 4 | [BBC News Classification](https://www.kaggle.com/competitions/learn-ai-bbc) | 文字・5 類 | TF-IDF + Linear SVM | `Choice`：business / entertainment / politics / sport / tech |
| 5 | [Iris](https://www.kaggle.com/datasets/uciml/iris) | 數值・3 類 | StandardScaler + Logistic Regression | `Choice`：setosa / versicolor / virginica |

這 5 題刻意混合了 Jev 應該擅長的文字判斷（2–4），以及 Jev 較不擅長、偏數值規則的表格題（1、5），
這樣才看得出 Jev 適合用在哪裡。

## 實驗設計

- **切分**：每個資料集做分層 80/20 切分（`random_state=42`）。baseline 只用 80% 訓練。
- **Jev 測試子集**：從 20% 測試集中分層抽出最多 `--jev-n` 筆（預設 200；Titanic 179 筆、Iris 30 筆則全用），
  baseline 與 Jev 都在**同一個子集**上評分，baseline 另外也報告完整測試集的分數。
- **Jev 是零樣本**：Jev 看不到任何訓練標籤，只收到單筆資料的 `state` 和一個有型別的問題。
  - 二分類用 `Noul`，取 `noul ≥ 0.5` 為正類，並用 `noul` 機率算 ROC AUC。
  - 多分類用 `Choice`，取 `choice` 為預測，另外統計 `confidence ≥ 0.8` 時的準確率與覆蓋率。
- **避免作弊**：
  - Titanic 不送乘客全名，只送稱謂（Mr/Mrs/Miss…），以免 Jev 靠記憶認出真實人物。
  - Iris 的選項描述只有物種名稱，不寫花瓣長度門檻，否則等於是我們自己寫規則，而不是 Jev 的判斷。
- **指標**：accuracy、macro-F1；二分類另有正類 F1 與 ROC AUC；Jev 另記錄 p50/p95 延遲與 token 用量。

程式結構：

```
jev_bench/datasets.py    下載（Kaggle 檔案的 GitHub 鏡像）、載入、分層切分
jev_bench/tasks.py       每題的 baseline pipeline 與 Jev 的 state / question / 解碼
jev_bench/jev_runner.py  非同步呼叫 Jev（可設並行數），答案快取在 .cache/jev/，重跑不重複計費
run_benchmark.py         CLI，輸出 results/<task>.json 與 results/summary.md
```

## 執行

```bash
pip install -r requirements.txt

python run_benchmark.py --mode baseline          # 只跑 Kaggle baseline，不需要 API key
python run_benchmark.py --mode dry-run           # 輸出每題送給 Jev 的範例請求 results/*_request_example.json

export TYPESAFE_API_KEY=...                       # 在 https://typesafe.ai 取得
python run_benchmark.py --mode jev --jev-n 200   # 跑 Jev，結果合併進 results/ 並更新 summary.md
python run_benchmark.py --mode both --tasks sms_spam imdb --concurrency 16
```

官方 Python SDK 是 [`typesafe-sdk`](https://pypi.org/project/typesafe-sdk/)（`import typesafe_sdk`），
不是 `typesafe` 或 `typesafe-ai`（後兩者分別是無關套件與防搶註的轉址套件）。

## 目前結果

Baseline 已跑完（見 [`results/summary.md`](results/summary.md)）：

| 題目 | Baseline accuracy（Jev 子集） | Baseline accuracy（完整測試集） | Jev |
| --- | --- | --- | --- |
| Titanic | 0.821 | 0.821 | 待跑 |
| SMS Spam | 0.980 | 0.985 | 待跑 |
| IMDB | 0.910 | 0.918 | 待跑 |
| BBC News | 0.990 | 0.987 | 待跑 |
| Iris | 0.933 | 0.933 | 待跑 |

**Jev 的數字還沒有**：建立這個 repo 的雲端環境網路政策擋住了 `api.typesafe.ai`，也沒有設定 `TYPESAFE_API_KEY`。
程式的 Jev 路徑已用假 key 測過（連線失敗會被記錄成 error、不會寫進快取，重跑會自動補送）。
只要在有網路與 API key 的環境執行 `python run_benchmark.py --mode jev` 即可補上。
