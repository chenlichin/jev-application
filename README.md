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
- **測試集**：下方結果中，所有方法都在**完整的 20% 測試集**上評分（`--jev-n 0`），
  也就是和 Kaggle 模型考同一份、訓練時沒看過的考卷。
  想省呼叫次數時可用 `--jev-n 200` 只抽分層子集，報告會多一欄 Kaggle 在完整測試集的分數，用來檢查子集是否有代表性。
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
- **Stacking（`--mode stack`）**：把 Jev 的機率當成特徵，和 Kaggle 模型的分數一起餵給第二階段的邏輯迴歸。
  - 第一階段：Kaggle 模型以 5-fold 交叉驗證在訓練集上產生 out-of-fold 分數（每筆的分數都來自沒看過它的模型）；
    測試集分數來自用整個訓練集訓練的模型。
  - Jev：在最多 2,000 筆分層抽樣的訓練資料上取得機率（Titanic、Iris 用全部訓練資料），測試集沿用快取。
    用 cluster-shot 時，被當成例子的訓練列不會進入第二階段，避免洩漏。
  - 第二階段：`StandardScaler + LogisticRegression`，輸入是兩者的 logit（多分類是每個類別一欄）。
  - **對照組**：同樣的第二階段只用 Kaggle 分數（記在 `stage2_kaggle_only`）。
    重新訓練本身會改變決策門檻，比較 Jev 的貢獻要和這個對照組比。
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
jev_bench/stacking.py    Kaggle out-of-fold 分數 + Jev 機率的第二階段模型，以及只用 Kaggle 分數的對照組
jev_bench/jev_runner.py  非同步呼叫 Jev（可設並行數），答案快取在 .cache/jev/，重跑不重複計費
run_benchmark.py         CLI，輸出 results/<task>.json 與 results/summary.md
```

## 執行

```bash
pip install -r requirements.txt

python run_benchmark.py --mode baseline          # 只跑 Kaggle baseline，不需要 API key
python run_benchmark.py --mode dry-run           # 輸出每題送給 Jev 的範例請求 results/*_request_example.json

export TYPESAFE_API_KEY=...                       # 在 https://typesafe.ai 取得
python run_benchmark.py --mode jev --jev-n 0     # 跑 Jev（完整測試集），結果合併進 results/ 並更新 summary.md
python run_benchmark.py --mode jev --jev-n 200   # 只抽 200 筆分層子集，省呼叫次數
python run_benchmark.py --mode jev --shots 0 3   # 同時跑 0-shot 與每類 3 個例子的 3-shot
python run_benchmark.py --mode jev --shots cluster cluster-random   # 分群挑例子 + 數量相同的隨機對照
python run_benchmark.py --mode stack --concurrency 32   # Kaggle + Jev stacking（0-shot 與 cluster 兩種特徵）
python run_benchmark.py --mode both --tasks sms_spam imdb --concurrency 16
```

官方 Python SDK 是 [`typesafe-sdk`](https://pypi.org/project/typesafe-sdk/)（`import typesafe_sdk`），
不是 `typesafe` 或 `typesafe-ai`（後兩者分別是無關套件與防搶註的轉址套件）。

## 結果（model `jev-latest` = `jev-1.13.0`，完整測試集）

完整數字見 [`results/summary.md`](results/summary.md)（每次執行自動重建），分成四張表：
**Summary**（資料集、筆數、要解決的問題、Jev 問題定義、第一筆送進 Jev 的 state）、**Accuracy**、**Macro-F1**、
**ROC AUC**（僅二分類題）；每張指標表列出所有方法，並把最佳分數標成粗體加底線。

所有方法都在完整 20% 測試集上評分（Kaggle 模型用 80% 訓練，Jev 例子也只從 80% 挑）：

| 題目 | n | Accuracy · Kaggle | Accuracy · Jev 0-shot | Accuracy · Jev 3-shot | Accuracy · Jev cluster | Accuracy · 隨機對照 | AUC · Kaggle | AUC · Jev 0-shot |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IMDB 影評 | 10,000 | 0.918 | **0.963** | **0.963** | **0.963** | **0.963** | 0.975 | **0.993** |
| SMS Spam | 1,115 | **0.985** | 0.979 | 0.980 | 0.980 | 0.981 | **0.992** | 0.991 |
| BBC News | 445 | **0.987** | 0.982 | 0.978 | 0.982 | 0.975 | — | — |
| Titanic | 179 | **0.821** | 0.648 | 0.682 | 0.754 | 0.682 | **0.841** | 0.749 |
| Iris | 30 | 0.933 | 0.600 | 0.900 | **0.967** | 0.833 | — | — |

約 5 萬次呼叫：p50 延遲約 0.29 秒、p95 約 0.45–0.63 秒；2 次暫時失敗在重跑時自動補送成功，最終 0 失敗。

### 觀察

- **IMDB：Jev 0-shot 在 10,000 筆上贏 Kaggle 4.5 個百分點（0.963 vs 0.918），AUC 0.993 vs 0.975。**
  Kaggle 模型用了 40,000 筆訓練資料，Jev 一筆都沒用。這是整份報告最可信的結論（樣本數最大）。
- **SMS、BBC：打平。** SMS 準確率差 0.6 個百分點、AUC 幾乎一樣（0.991 vs 0.992）；
  BBC 在 445 篇中少對 2 篇。BBC 有 93.9% 的答案信心 ≥ 0.8，這部分準確率 99.3%，信心可用來決定哪些要人工複查。
- **Titanic：Jev 沒有抓到「婦孺優先」。** 0-shot 時女性平均生還機率只給 0.40（實際 0.74），男性 0.33（實際 0.20），
  幾乎沒有區分性別；機率整體偏低，門檻 0.5 時只預測 25% 生還（實際 38%）。
- **Iris：0-shot 時 Jev 從來沒有預測 virginica**（10 朵全被判為 versicolor）。
  純數值、需要從資料學邊界的問題不是 Jev 的強項。

### Few-shot 的效果

- **文字題：例子沒有幫助。** 在完整測試集上 IMDB 四種 Jev 方法都是 0.963、SMS 在 0.979–0.981 之間，
  BBC 3-shot 反而少對 2 篇；token 成本卻是 0-shot 的 2–3.5 倍（IMDB：620 萬 → 1,435 萬 token）。
- **表格／數值題：例子有明顯幫助**，Iris 0.60 → 0.90、Titanic 0.648 → 0.682（每類 3 個隨機例子）。

### Cluster few-shot：用分群挑「邊緣 pattern」

| 題目 | k | 例子數（每類） | Accuracy · 0-shot | Accuracy · 3-shot | Accuracy · **cluster** | Accuracy · 隨機對照（同數量） | Accuracy · Kaggle |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Titanic | 3 | 3 / 3 | 0.648 | 0.682 | **0.754** | 0.682 | 0.821 |
| Iris | 3 | 1 / 2 / 2 | 0.600 | 0.900 | **0.967** | 0.833 | 0.933 |
| BBC News | 8 | 3 / 4 / 2 / 3 / 3 | 0.982 | 0.978 | 0.982 | 0.975 | 0.987 |
| SMS Spam | 8 | 8 / 5 | 0.979 | 0.980 | 0.980 | 0.981 | 0.985 |
| IMDB | 2 | 2 / 2 | 0.963 | 0.963 | 0.963 | 0.963 | 0.918 |

- **Titanic：0.682 → 0.754，AUC 0.783 → 0.826。** 例子數完全相同（每類 3 個、token 幾乎一樣），
  唯一差別是挑法：分群挑到了「生還者中的少數群」（生還者只佔該群 25%）與「死者中的少數群」（死者只佔該群 30%）。
  隨機對照組剛好與 3-shot 抽到同一組例子（同數量、同 seed），所以兩者分數相同；179 筆中 cluster 多對 13 筆。
- **Iris：0.900 → 0.967。** 只用 5 個例子（3-shot 用 9 個）就更好；同數量的隨機對照只有 0.833，
  所以提升來自多樣性而不是例子數。分群挑到的是 5 朵「長得像 virginica 的 versicolor」
  與 14 朵「長得像 versicolor 的 virginica」這兩個混淆區的代表。
  Iris 只有 30 筆，0.967 vs Kaggle 0.933 只差 1 朵，不能說贏過 Kaggle。
- **文字題沒有差別。** 高維文字的 silhouette 只有 0.02–0.08，分群本身就不太明確
  （IMDB 只分出 2 群，SMS／BBC 都頂到上限 8 群）。

### Stacking：把 Jev 機率當特徵加進 Kaggle 模型

| 題目 | n | Accuracy · Kaggle | Accuracy · Jev 0-shot | Accuracy · 第二階段只用 Kaggle（對照） | Accuracy · Kaggle + Jev 0-shot | Accuracy · Kaggle + Jev cluster | AUC · 對照 | AUC · Kaggle + Jev 0-shot | AUC · Kaggle + Jev cluster |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SMS Spam | 1,115 | 0.985 | 0.979 | 0.985 | **0.992** | 0.991 | 0.992 | **0.994** | 0.993 |
| BBC News | 445 | 0.987 | 0.982 | 0.987 | 0.989 | **0.991** | — | — | — |
| IMDB | 10,000 | 0.918 | 0.963 | 0.918 | **0.964** | **0.964** | 0.975 | **0.994** | **0.994** |
| Titanic | 179 | **0.821** | 0.648 | 0.810 | 0.793 | 0.799 | 0.841 | 0.847 | **0.859** |
| Iris | 30 | 0.933 | 0.600 | 0.967 | 0.967 | **1.000** | — | — | — |

- **SMS：兩者互補，得到全場最佳。** 0.992 高於 Kaggle 單獨的 0.985 與 Jev 單獨的 0.979，
  1,115 封中比 Kaggle 多對 8 封；第二階段給 Jev 的權重（3.07）比 Kaggle（1.89）還高。
- **BBC：小幅提升**，0.987 → 0.991（445 篇中多對 2 篇）。
- **IMDB：幾乎等於 Jev 單獨**（0.963 → 0.964）。第二階段給 Jev 的權重約是 Kaggle 的 4 倍，
  Kaggle 模型能補的資訊很少。
- **Titanic：排序變好、準確率沒變好。** AUC 0.841 → 0.859，但 accuracy 比對照組少 2 位乘客；
  第二階段仍主要依賴 Kaggle 分數（權重 1.67 vs Jev 0.56）。
- **Iris：** cluster 特徵讓 30 朵全對，但只比對照組多 1 朵。注意光是重新訓練第二階段就從 0.933 變成 0.967。

### 結論

- **文字分類：直接用 Jev 0-shot；要榨出最後一點準確率就做 stacking。**
  SMS、BBC 的 stacking 都超過兩者單獨的表現。 IMDB 明顯勝過 Kaggle 解法，SMS、BBC 打平，而且完全不需要訓練資料；
  加例子只會增加成本。
- **表格／數值題：Kaggle 傳統模型仍然較好。** 若要用 Jev，務必加例子，而且用「分群挑邊緣例子」比隨機挑更好；
  或把 Jev 的判斷當成額外特徵交給傳統模型。

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
