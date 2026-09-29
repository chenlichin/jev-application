# Kaggle × Jev benchmark results

## Pipeline overview

`n` is the stratified test subset both methods are scored on. Kaggle columns are the classic
Kaggle solution trained on the 80% train split; Jev is zero-shot and never sees training labels.

<table>
<thead>
<tr>
<th rowspan="2">Dataset</th><th rowspan="2">n</th><th rowspan="2">Problem</th>
<th rowspan="2">Jev question</th><th rowspan="2">State (first row)</th>
<th colspan="3">Accuracy</th><th colspan="2">Macro-F1</th>
</tr>
<tr>
<th>Kaggle</th><th>Jev zero-shot</th><th>Kaggle full test</th>
<th>Kaggle</th><th>Jev zero-shot</th>
</tr>
</thead>
<tbody>
<tr>
<td><a href="https://www.kaggle.com/competitions/titanic">titanic</a></td>
<td>179<br><sub>full test 179<br>train 712</sub></td>
<td>Predict whether a Titanic passenger survived</td>
<td><b>Noul</b>: <code>passenger</code> describes a person aboard the RMS Titanic when it sank on 15 April 1912. Based on this profile and what is known about who reached the lifeboats, did this passenger survive?<br><sub>answers: yes / no</sub></td>
<td><code>{&quot;passenger&quot;: {&quot;ticket_class&quot;: &quot;3rd (lower)&quot;, &quot;sex&quot;: &quot;male&quot;, &quot;title&quot;: &quot;Mr&quot;, &quot;age_years&quot;: 24.0, &quot;siblings_or_spouses_aboard&quot;: 2, &quot;parents_or_children_aboard&quot;: 0, …</code><br><sub>true label: 0</sub></td>
<td>0.821</td><td>0.648</td><td>0.821</td>
<td>0.810</td><td>0.593</td>
</tr>
<tr>
<td><a href="https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset">sms_spam</a></td>
<td>200<br><sub>full test 1115<br>train 4457</sub></td>
<td>Detect whether an SMS message is spam</td>
<td><b>Noul</b>: Is <code>sms</code> a spam text message?<br><sub>answers: yes / no</sub></td>
<td><code>{&quot;sms&quot;: &quot;Aight I&#x27;ll grab something to eat too, text me when you&#x27;re back at mu&quot;}</code><br><sub>true label: 0</sub></td>
<td>0.980</td><td>0.975</td><td>0.985</td>
<td>0.954</td><td>0.947</td>
</tr>
<tr>
<td><a href="https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews">imdb</a></td>
<td>200<br><sub>full test 10000<br>train 40000</sub></td>
<td>Classify a movie review as positive or negative</td>
<td><b>Noul</b>: Is <code>review</code> a positive review overall, meaning the reviewer recommends the movie?<br><sub>answers: yes / no</sub></td>
<td><code>{&quot;review&quot;: &quot;Without reiterating what was said above about this movie, I would like to add that I was looking forward to watching this film...the cast/location a …</code><br><sub>true label: 0</sub></td>
<td>0.910</td><td>0.950</td><td>0.918</td>
<td>0.910</td><td>0.950</td>
</tr>
<tr>
<td><a href="https://www.kaggle.com/competitions/learn-ai-bbc">bbc_news</a></td>
<td>200<br><sub>full test 445<br>train 1780</sub></td>
<td>Assign a BBC News article to one of 5 sections</td>
<td><b>Choice</b>: Which BBC News section does <code>article</code> belong to?<br><sub>answers: business / entertainment / politics / sport / tech</sub></td>
<td><code>{&quot;article&quot;: &quot;ministers  naive  over phone-taps the government is being naive by refusing to allow phone-tap evidence in court  a senior eu politician says.  jav …</code><br><sub>true label: politics</sub></td>
<td>0.990</td><td>0.985</td><td>0.987</td>
<td>0.990</td><td>0.985</td>
</tr>
<tr>
<td><a href="https://www.kaggle.com/datasets/uciml/iris">iris</a></td>
<td>30<br><sub>full test 30<br>train 120</sub></td>
<td>Identify the iris species from 4 flower measurements</td>
<td><b>Choice</b>: <code>flower_measurements_cm</code> are measurements of one iris flower. Which species is it?<br><sub>answers: setosa / versicolor / virginica</sub></td>
<td><code>{&quot;flower_measurements_cm&quot;: {&quot;sepal_length&quot;: 4.4, &quot;sepal_width&quot;: 3.0, &quot;petal_length&quot;: 1.3, &quot;petal_width&quot;: 0.2}}</code><br><sub>true label: setosa</sub></td>
<td>0.933</td><td>0.600</td><td>0.933</td>
<td>0.933</td><td>0.494</td>
</tr>
</tbody>
</table>

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
