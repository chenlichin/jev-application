# Kaggle × Jev benchmark results

`n` is the stratified test subset every method is scored on; "Kaggle full test" is the whole 20% test split.
Kaggle is the classic Kaggle solution trained on the 80% train split. Jev 0-shot never sees training labels;
Jev k-shot adds k random labeled training examples per class to each answer's criteria; Jev cluster-shot takes
one example per (label, k-means cluster) cell; Jev matched random is its control with the same per-label counts.
Examples always come from the train split. Bold marks the best score on the shared subset.

## 1. Summary

| Dataset | n | Train | Problem | Jev question | State (first row) | True label |
| --- | ---: | ---: | --- | --- | --- | --- |
| [titanic](https://www.kaggle.com/competitions/titanic) | 179 | 712 | Predict whether a Titanic passenger survived | **Noul**: `passenger` describes a person aboard the RMS Titanic when it sank on 15 April 1912. Based on this profile and what is known about who reached the lifeboats, did this passenger survive? (yes / no) | `{"passenger": {"ticket_class": "3rd (lower)", "sex": "male", "title": "Mr", "age_years": 24.0, "siblings_or_sp …` | 0 |
| [sms_spam](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) | 200 | 4457 | Detect whether an SMS message is spam | **Noul**: Is `sms` a spam text message? (yes / no) | `{"sms": "Aight I'll grab something to eat too, text me when you're back at mu"}` | 0 |
| [imdb](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) | 200 | 40000 | Classify a movie review as positive or negative | **Noul**: Is `review` a positive review overall, meaning the reviewer recommends the movie? (yes / no) | `{"review": "Without reiterating what was said above about this movie, I would like to add that I was looking f …` | 0 |
| [bbc_news](https://www.kaggle.com/competitions/learn-ai-bbc) | 200 | 1780 | Assign a BBC News article to one of 5 sections | **Choice**: Which BBC News section does `article` belong to? (business / entertainment / politics / sport / tech) | `{"article": "ministers naive over phone-taps the government is being naive by refusing to allow phone-tap ev …` | politics |
| [iris](https://www.kaggle.com/datasets/uciml/iris) | 30 | 120 | Identify the iris species from 4 flower measurements | **Choice**: `flower_measurements_cm` are measurements of one iris flower. Which species is it? (setosa / versicolor / virginica) | `{"flower_measurements_cm": {"sepal_length": 4.4, "sepal_width": 3.0, "petal_length": 1.3, "petal_width": 0.2}}` | setosa |

## 2. Accuracy

| Dataset | n | Kaggle | Jev 0-shot | Jev 3-shot | Jev cluster-shot | Jev matched random | Kaggle full test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| titanic | 179 | **0.821** | 0.648 | 0.682 | 0.754 | 0.682 | 0.821 |
| sms_spam | 200 | **0.980** | 0.975 | 0.975 | 0.975 | 0.975 | 0.985 |
| imdb | 200 | 0.910 | **0.950** | **0.950** | **0.950** | **0.950** | 0.918 |
| bbc_news | 200 | **0.990** | 0.985 | 0.980 | 0.985 | 0.980 | 0.987 |
| iris | 30 | 0.933 | 0.600 | 0.900 | **0.967** | 0.833 | 0.933 |

## 3. Macro-F1

| Dataset | n | Kaggle | Jev 0-shot | Jev 3-shot | Jev cluster-shot | Jev matched random | Kaggle full test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| titanic | 179 | **0.810** | 0.593 | 0.644 | 0.734 | 0.644 | 0.810 |
| sms_spam | 200 | **0.954** | 0.947 | 0.947 | 0.947 | 0.947 | 0.965 |
| imdb | 200 | 0.910 | **0.950** | **0.950** | **0.950** | **0.950** | 0.918 |
| bbc_news | 200 | **0.990** | 0.985 | 0.980 | 0.985 | 0.980 | 0.986 |
| iris | 30 | 0.933 | 0.494 | 0.898 | **0.967** | 0.828 | 0.933 |

## 4. ROC AUC (binary tasks)

Uses the Kaggle model's positive-class score and Jev's `noul` probability.

| Dataset | n | Kaggle | Jev 0-shot | Jev 3-shot | Jev cluster-shot | Jev matched random | Kaggle full test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| titanic | 179 | **0.841** | 0.749 | 0.783 | 0.826 | 0.783 | 0.841 |
| sms_spam | 200 | 0.967 | 0.981 | **0.982** | 0.974 | 0.974 | 0.992 |
| imdb | 200 | 0.979 | **0.991** | **0.991** | 0.990 | 0.990 | 0.975 |

## Run details

- **titanic** Jev 0-shot: `{"model": "jev-latest", "shots": "0", "errors": 0, "latency_p50_s": 0.287, "latency_p95_s": 0.364, "input_tokens_total": 80056}`
- **titanic** Jev 3-shot: `{"model": "jev-latest", "shots": "3", "errors": 0, "examples_per_label": {"0": 3, "1": 3}, "latency_p50_s": 0.273, "latency_p95_s": 0.512, "input_tokens_total": 201955}`
- **titanic** Jev cluster-shot: `{"model": "jev-latest", "shots": "cluster", "errors": 0, "examples_per_label": {"0": 3, "1": 3}, "latency_p50_s": 0.28, "latency_p95_s": 0.437, "input_tokens_total": 202492}`
- **titanic** Jev matched random: `{"model": "jev-latest", "shots": "cluster-random", "errors": 0, "examples_per_label": {"0": 3, "1": 3}, "latency_p50_s": 0.273, "latency_p95_s": 0.512, "input_tokens_total": 201955}`
- **sms_spam** Jev 0-shot: `{"model": "jev-latest", "shots": "0", "errors": 0, "latency_p50_s": 0.286, "latency_p95_s": 0.46, "input_tokens_total": 75034}`
- **sms_spam** Jev 3-shot: `{"model": "jev-latest", "shots": "3", "errors": 0, "examples_per_label": {"0": 3, "1": 3}, "latency_p50_s": 0.275, "latency_p95_s": 0.367, "input_tokens_total": 149634}`
- **sms_spam** Jev cluster-shot: `{"model": "jev-latest", "shots": "cluster", "errors": 0, "examples_per_label": {"0": 8, "1": 5}, "latency_p50_s": 0.285, "latency_p95_s": 0.378, "input_tokens_total": 177434}`
- **sms_spam** Jev matched random: `{"model": "jev-latest", "shots": "cluster-random", "errors": 0, "examples_per_label": {"0": 8, "1": 5}, "latency_p50_s": 0.284, "latency_p95_s": 0.387, "input_tokens_total": 200634}`
- **imdb** Jev 0-shot: `{"model": "jev-latest", "shots": "0", "errors": 0, "latency_p50_s": 0.286, "latency_p95_s": 0.439, "input_tokens_total": 121414}`
- **imdb** Jev 3-shot: `{"model": "jev-latest", "shots": "3", "errors": 0, "examples_per_label": {"0": 3, "1": 3}, "latency_p50_s": 0.279, "latency_p95_s": 0.378, "input_tokens_total": 283814}`
- **imdb** Jev cluster-shot: `{"model": "jev-latest", "shots": "cluster", "errors": 0, "examples_per_label": {"0": 2, "1": 2}, "latency_p50_s": 0.283, "latency_p95_s": 0.427, "input_tokens_total": 243614}`
- **imdb** Jev matched random: `{"model": "jev-latest", "shots": "cluster-random", "errors": 0, "examples_per_label": {"0": 2, "1": 2}, "latency_p50_s": 0.284, "latency_p95_s": 0.371, "input_tokens_total": 229814}`
- **bbc_news** Jev 0-shot: `{"model": "jev-latest", "shots": "0", "errors": 0, "latency_p50_s": 0.287, "latency_p95_s": 0.358, "input_tokens_total": 181042, "accuracy_when_confidence_ge_0.8": 0.9946524064171123, "share_confidence_ge_0.8": 0.935}`
- **bbc_news** Jev 3-shot: `{"model": "jev-latest", "shots": "3", "errors": 0, "examples_per_label": {"business": 3, "entertainment": 3, "politics": 3, "sport": 3, "tech": 3}, "latency_p50_s": 0.282, "latency_p95_s": 0.405, "input_tokens_total": 632842, "accuracy_when_confidence_ge_0.8": 0.9946236559139785, "share_confidence_ge_0.8": 0.93}`
- **bbc_news** Jev cluster-shot: `{"model": "jev-latest", "shots": "cluster", "errors": 0, "examples_per_label": {"business": 3, "entertainment": 4, "politics": 2, "sport": 3, "tech": 3}, "latency_p50_s": 0.291, "latency_p95_s": 0.38, "input_tokens_total": 615042, "accuracy_when_confidence_ge_0.8": 0.9946808510638298, "share_confidence_ge_0.8": 0.94}`
- **bbc_news** Jev matched random: `{"model": "jev-latest", "shots": "cluster-random", "errors": 0, "examples_per_label": {"business": 3, "entertainment": 4, "politics": 2, "sport": 3, "tech": 3}, "latency_p50_s": 0.284, "latency_p95_s": 0.345, "input_tokens_total": 634442, "accuracy_when_confidence_ge_0.8": 0.9946236559139785, "share_confidence_ge_0.8": 0.93}`
- **iris** Jev 0-shot: `{"model": "jev-latest", "shots": "0", "errors": 0, "latency_p50_s": 0.349, "latency_p95_s": 0.734, "input_tokens_total": 11910, "accuracy_when_confidence_ge_0.8": 1.0, "share_confidence_ge_0.8": 0.3333333333333333}`
- **iris** Jev 3-shot: `{"model": "jev-latest", "shots": "3", "errors": 0, "examples_per_label": {"setosa": 3, "versicolor": 3, "virginica": 3}, "latency_p50_s": 0.281, "latency_p95_s": 0.581, "input_tokens_total": 27300, "accuracy_when_confidence_ge_0.8": 0.9565217391304348, "share_confidence_ge_0.8": 0.7666666666666667}`
- **iris** Jev cluster-shot: `{"model": "jev-latest", "shots": "cluster", "errors": 0, "examples_per_label": {"setosa": 1, "versicolor": 2, "virginica": 2}, "latency_p50_s": 0.286, "latency_p95_s": 0.57, "input_tokens_total": 20940, "accuracy_when_confidence_ge_0.8": 1.0, "share_confidence_ge_0.8": 0.8}`
- **iris** Jev matched random: `{"model": "jev-latest", "shots": "cluster-random", "errors": 0, "examples_per_label": {"setosa": 1, "versicolor": 2, "virginica": 2}, "latency_p50_s": 0.277, "latency_p95_s": 0.591, "input_tokens_total": 20940, "accuracy_when_confidence_ge_0.8": 0.9565217391304348, "share_confidence_ge_0.8": 0.7666666666666667}`
