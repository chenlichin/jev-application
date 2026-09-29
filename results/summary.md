# Kaggle × Jev benchmark results

## Pipeline overview

`n` is the stratified test subset both methods are scored on; "Kaggle full test" is the whole 20% test split.
Kaggle columns are the classic Kaggle solution trained on the 80% train split; Jev is zero-shot and never
sees training labels. Row counts for every split are in `results/<task>.json` under `pipeline`.

| Dataset | n | Problem | Jev question | State (first row) | True label | Accuracy · Kaggle | Accuracy · Jev | Accuracy · Kaggle full test | Macro-F1 · Kaggle | Macro-F1 · Jev |
| --- | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| [titanic](https://www.kaggle.com/competitions/titanic) | 179 | Predict whether a Titanic passenger survived | **Noul**: `passenger` describes a person aboard the RMS Titanic when it sank on 15 April 1912. Based on this profile and what is known about who reached the lifeboats, did this passenger survive? (yes / no) | `{"passenger": {"ticket_class": "3rd (lower)", "sex": "male", "title": "Mr", "age_years": 24.0, "siblings_or_sp …` | 0 | 0.821 | 0.648 | 0.821 | 0.810 | 0.593 |
| [sms_spam](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) | 200 | Detect whether an SMS message is spam | **Noul**: Is `sms` a spam text message? (yes / no) | `{"sms": "Aight I'll grab something to eat too, text me when you're back at mu"}` | 0 | 0.980 | 0.975 | 0.985 | 0.954 | 0.947 |
| [imdb](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) | 200 | Classify a movie review as positive or negative | **Noul**: Is `review` a positive review overall, meaning the reviewer recommends the movie? (yes / no) | `{"review": "Without reiterating what was said above about this movie, I would like to add that I was looking f …` | 0 | 0.910 | 0.950 | 0.918 | 0.910 | 0.950 |
| [bbc_news](https://www.kaggle.com/competitions/learn-ai-bbc) | 200 | Assign a BBC News article to one of 5 sections | **Choice**: Which BBC News section does `article` belong to? (business / entertainment / politics / sport / tech) | `{"article": "ministers naive over phone-taps the government is being naive by refusing to allow phone-tap ev …` | politics | 0.990 | 0.985 | 0.987 | 0.990 | 0.985 |
| [iris](https://www.kaggle.com/datasets/uciml/iris) | 30 | Identify the iris species from 4 flower measurements | **Choice**: `flower_measurements_cm` are measurements of one iris flower. Which species is it? (setosa / versicolor / virginica) | `{"flower_measurements_cm": {"sepal_length": 4.4, "sepal_width": 3.0, "petal_length": 1.3, "petal_width": 0.2}}` | setosa | 0.933 | 0.600 | 0.933 | 0.933 | 0.494 |

## All metrics

Same stratified test subset for both methods (`jev_subset`). The baseline also reports the full 20% test split.

| Task | Metric | Kaggle baseline (subset) | Jev zero-shot (subset) | Baseline (full test) |
| --- | --- | --- | --- | --- |
| titanic | accuracy | 0.8212 | 0.6480 | 0.8212 |
| titanic | macro_f1 | 0.8103 | 0.5927 | 0.8103 |
| titanic | roc_auc | 0.8412 | 0.7488 | 0.8412 |
| sms_spam | accuracy | 0.9800 | 0.9750 | 0.9848 |
| sms_spam | macro_f1 | 0.9543 | 0.9473 | 0.9654 |
| sms_spam | roc_auc | 0.9675 | 0.9805 | 0.9922 |
| imdb | accuracy | 0.9100 | 0.9500 | 0.9180 |
| imdb | macro_f1 | 0.9100 | 0.9500 | 0.9180 |
| imdb | roc_auc | 0.9787 | 0.9906 | 0.9747 |
| bbc_news | accuracy | 0.9900 | 0.9850 | 0.9865 |
| bbc_news | macro_f1 | 0.9896 | 0.9851 | 0.9861 |
| iris | accuracy | 0.9333 | 0.6000 | 0.9333 |
| iris | macro_f1 | 0.9333 | 0.4935 | 0.9333 |

- **titanic** Jev details: `{"model": "jev-latest", "errors": 0, "latency_p50_s": 0.287, "latency_p95_s": 0.364, "input_tokens_total": 80056}`
- **sms_spam** Jev details: `{"model": "jev-latest", "errors": 0, "latency_p50_s": 0.286, "latency_p95_s": 0.46, "input_tokens_total": 75034}`
- **imdb** Jev details: `{"model": "jev-latest", "errors": 0, "latency_p50_s": 0.286, "latency_p95_s": 0.439, "input_tokens_total": 121414}`
- **bbc_news** Jev details: `{"model": "jev-latest", "errors": 0, "latency_p50_s": 0.287, "latency_p95_s": 0.358, "input_tokens_total": 181042, "accuracy_when_confidence_ge_0.8": 0.9946524064171123, "share_confidence_ge_0.8": 0.935}`
- **iris** Jev details: `{"model": "jev-latest", "errors": 0, "latency_p50_s": 0.349, "latency_p95_s": 0.734, "input_tokens_total": 11910, "accuracy_when_confidence_ge_0.8": 1.0, "share_confidence_ge_0.8": 0.3333333333333333}`
