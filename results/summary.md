# Kaggle × Jev benchmark results

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
