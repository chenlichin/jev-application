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
- **Few-shot（`--shots k`）**：從**訓練集**每類抽 k 筆有標籤的例子（固定 seed，所有測試資料共用同一組），
  附加到該答案的 criteria 裡；instructions 不變，所以 k-shot 與 0-shot 只差在例子：
  ```json
  "criteria": {
    "true":  {"description": "The passenger survived the sinking.", "examples": [{"passenger": {...}}, ...]},
    "false": {"description": "The passenger died in the sinking.",  "examples": [...]}
  }
  ```
  優先挑短的例子（JSON ≤ 700 字）；沒有夠短的（BBC 幾乎全部）就把例子裡的文字截到前 600 字。
  用到的訓練列 index 記在 `results/<task>.json` 的 `shot_train_rows`，可重現、可確認沒有測試資料混入。
- **Cluster few-shot（`--shots cluster`）**：隨機抽樣多半抽到每類「最典型」的樣本。這裡先用 k-means 把訓練集分群
  （特徵與 baseline 相同：文字用 TF-IDF→SVD 100 維，表格用標準化數值＋one-hot），
  群數在 `類別數..8` 之間取 silhouette 最高者；再把**每個 y 類別與每個群取交集**，
  每個 ≥ 3 筆的交集格取一個代表例（最接近該格中心的樣本）。
  一個 y 分散在多個群 → 每個群都出一個例子，其中 y 在該群是少數的格子（`label_share_of_cluster < 0.5`）
  就是靠近決策邊界的「邊緣 pattern」。分群結果記在 `results/<task>.json` 的 `jev_cluster.clustering`。
- **對照組（`--shots cluster-random`）**：每類抽**與 cluster 相同數量**的隨機例子，
  用來區分「多樣性」與「單純例子變多」的效果。
- **避免作弊**：
  - Titanic 不送乘客全名，只送稱謂（Mr/Mrs/Miss…），以免 Jev 靠記憶認出真實人物。
  - Iris 的選項描述只有物種名稱，不寫花瓣長度門檻，否則等於是我們自己寫規則，而不是 Jev 的判斷。
- **指標**：accuracy、macro-F1；二分類另有正類 F1 與 ROC AUC；Jev 另記錄 p50/p95 延遲與 token 用量。

程式結構：

```
jev_bench/datasets.py    下載（Kaggle 檔案的 GitHub 鏡像）、載入、分層切分
jev_bench/tasks.py       每題的 baseline pipeline 與 Jev 的 state / question / 解碼
jev_bench/few_shot.py    從訓練集挑例子、把例子掛到 criteria 上
jev_bench/cluster_shots.py  k-means 分群 × 類別交集挑例子，以及數量相同的隨機對照組
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
python run_benchmark.py --mode jev --shots 0 3   # 同時跑 0-shot 與每類 3 個例子的 3-shot
python run_benchmark.py --mode jev --shots cluster cluster-random   # 分群挑例子 + 數量相同的隨機對照
python run_benchmark.py --mode both --tasks sms_spam imdb --concurrency 16
```

官方 Python SDK 是 [`typesafe-sdk`](https://pypi.org/project/typesafe-sdk/)（`import typesafe_sdk`），
不是 `typesafe` 或 `typesafe-ai`（後兩者分別是無關套件與防搶註的轉址套件）。

## 結果（model `jev-latest` = `jev-1.13.0`）

完整數字見 [`results/summary.md`](results/summary.md)（每次執行自動重建），分成四張表：
**Summary**（資料集、筆數、要解決的問題、Jev 問題定義、第一筆送進 Jev 的 state）、**Accuracy**、**Macro-F1**、
**ROC AUC**（僅二分類題）；每張指標表列出所有方法，並把同一子集上的最佳分數標成粗體加底線。
所有方法在同一個測試子集上比較：

| 題目 | 筆數 | Accuracy · Kaggle | Accuracy · Jev 0-shot | Accuracy · Jev 3-shot | AUC · Kaggle | AUC · Jev 0-shot | AUC · Jev 3-shot | Tokens · 0-shot → 3-shot |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| IMDB 影評 | 200 | 0.910 | **0.950** | **0.950** | 0.979 | 0.991 | **0.991** | 121k → 284k（×2.3） |
| SMS Spam | 200 | **0.980** | 0.975 | 0.975 | 0.967 | 0.981 | **0.982** | 75k → 150k（×2.0） |
| BBC News | 200 | **0.990** | 0.985 | 0.980 | — | — | — | 181k → 633k（×3.5） |
| Titanic | 179 | **0.821** | 0.648 | 0.682 | **0.841** | 0.749 | 0.783 | 80k → 202k（×2.5） |
| Iris | 30 | **0.933** | 0.600 | 0.900 | — | — | — | 12k → 27k（×2.3） |

每次呼叫 p50 延遲約 0.29 秒、p95 約 0.36–0.73 秒，1,000 次呼叫 0 失敗。

### 觀察

- **文字題（IMDB、SMS、BBC）：Jev 零樣本就與用上千、上萬筆訓練資料的 Kaggle 解法打平甚至更好。**
  IMDB 準確率高出 4 個百分點；SMS 的 AUC 更高；BBC 只差 1 筆。
  BBC 中 93.5% 的答案信心 ≥ 0.8，這部分準確率 99.5%，信心可以拿來決定哪些要人工複查。
- **Titanic：Jev 沒有抓到「婦孺優先」。** 女性平均生還機率只給 0.40（實際 0.74），男性 0.33（實際 0.20），
  幾乎沒有區分性別；機率整體偏低，門檻 0.5 時只預測 25% 生還（實際 38%）。
  AUC 0.75 表示排序有一定資訊，若門檻改為 0.4 準確率可到 0.73——但這個門檻是在測試集上看出來的，只能當診斷，不能當成績。
- **Iris：Jev 從來沒有預測 virginica。** 10 朵 virginica 全被判為 versicolor，setosa 則全對。
  純數值、需要從資料學邊界的問題不是 Jev 的用途；照 TypeSafe 的建議，這類規則應留在程式碼（或傳統模型）裡。

### Few-shot 的效果（每類 3 個例子）

- **Iris 大幅改善：0.60 → 0.90。** 0-shot 時 Jev 不知道三個品種的尺寸分界、從不預測 virginica；
  看了 9 朵有標籤的花之後，已接近 Kaggle 模型的 0.933（30 筆中只差 1 朵）。
- **Titanic 小幅改善：accuracy 0.648 → 0.682、AUC 0.749 → 0.783。** 例子讓性別差距拉開一些
  （女性平均生還機率 0.40 → 0.46、男性 0.33 → 0.30），但 6 個例子仍不足以學到「婦孺優先」的強度，離 0.821 還遠。
- **文字題幾乎不變**：IMDB、SMS 持平（AUC 微升），BBC 少對 1 筆（200 筆內的差異，屬於雜訊範圍）。
  0-shot 已經很好，例子帶來的資訊有限，token 成本卻變成 2–3.5 倍。

**建議**：文字題用 0-shot 就好；數值／表格題 few-shot 有明顯幫助，是值得付的 token 成本。

### Cluster few-shot：用分群挑「邊緣 pattern」

| 題目 | k | 例子數（每類） | Accuracy · 0-shot | Accuracy · 3-shot | Accuracy · **cluster** | Accuracy · 隨機對照（同數量） | Accuracy · Kaggle |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Titanic | 3 | 3 / 3 | 0.648 | 0.682 | **0.754** | 0.682 | 0.821 |
| Iris | 3 | 1 / 2 / 2 | 0.600 | 0.900 | **0.967** | 0.833 | 0.933 |
| BBC News | 8 | 3 / 4 / 2 / 3 / 3 | 0.985 | 0.980 | 0.985 | 0.980 | 0.990 |
| SMS Spam | 8 | 8 / 5 | 0.975 | 0.975 | 0.975 | 0.975 | 0.980 |
| IMDB | 2 | 2 / 2 | 0.950 | 0.950 | 0.950 | 0.950 | 0.910 |

- **Titanic：0.682 → 0.754，AUC 0.783 → 0.826。** 例子數完全相同（每類 3 個、token 幾乎一樣），
  唯一差別是挑法：分群挑到了「生還者中的少數群」（生還者只佔該群 25%）與「死者中的少數群」（死者只佔該群 30%）。
  隨機對照組剛好與 3-shot 抽到同一組例子（同數量、同 seed），所以兩者分數相同；179 筆中 cluster 多對 13 筆。
- **Iris：0.900 → 0.967，比 Kaggle 的 0.933 還高 1 朵。** 只用 5 個例子（3-shot 用 9 個）就更好；
  同數量的隨機對照只有 0.833，所以提升來自多樣性而不是例子數。
  分群挑到的是 5 朵「長得像 virginica 的 versicolor」與 14 朵「長得像 versicolor 的 virginica」這兩個混淆區的代表。
  注意 Iris 只有 30 筆，差 1 朵就是 3.3 個百分點。
- **文字題沒有差別。** 0-shot 已接近上限；此外高維文字的 silhouette 只有 0.02–0.08，分群本身就不太明確
  （IMDB 只分出 2 群，SMS／BBC 都頂到上限 8 群）。

**結論**：在表格／數值題，「用分群挑邊緣例子」比隨機挑同樣數量的例子明顯更好，是這次實驗最有效的改進；
文字題則不需要。

### 結論

Jev 適合取代「需要理解語意」的文字分類模型，而且不需要訓練資料；
表格／數值題靠 few-shot 可以縮小差距（Iris 幾乎追平），但目前仍以傳統模型為佳，或把 Jev 的判斷當成額外特徵交給傳統模型。

### 下一步可以試

1. **混合模型**：把 Jev 的 `noul` 機率當成特徵加進 Titanic 的梯度提升樹，看是否超過單獨的 baseline。
2. **門檻校正**：用訓練集（而非測試集）挑 Jev 的決策門檻。
3. **更多例子 / 動態例子**：`--shots 10` 看 Titanic 能否繼續進步；或改成針對每筆測試資料，從訓練集挑最相似的例子（kNN few-shot）。
   Cluster 版本也可以把 `MAX_K` 調大，讓 Titanic 分出更細的子群。
4. `--model jev-preview` 比較預覽版。
5. `--jev-n 0` 跑完整測試集，縮小 200 筆子集的誤差範圍。

## 在 Claude Code 雲端環境執行

API key 以 Bearer credential 存在環境設定（Allowed website `api.typesafe.ai`），由代理在送出時注入，
容器內看不到 key。SDK 仍要求 `TYPESAFE_API_KEY` 有值，所以隨便給一個佔位值即可：

```bash
TYPESAFE_API_KEY=injected-by-proxy python run_benchmark.py --mode jev --shots 0 3 cluster cluster-random
```
