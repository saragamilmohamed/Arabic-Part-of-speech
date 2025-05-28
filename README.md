
# Arabic Part-of-Speech Tagging with Transformers

Arabic Part-of-Speech Tagging with Transformers is an end-to-end pipeline that fine-tunes a BERT-style model to label each word in an Arabic sentence with its Universal Dependencies part-of-speech tag. Starting from raw CoNLL-U formatted treebank data:

- Loading and parsing a `.conllu` dataset with **pyconll**  
- Splitting into train / validation / test sets  
- Tokenizing with **asafaya/bert-base-arabic** (DistilBERT-style)  
- Aligning word-level POS tags to subword tokens  
- Fine-tuning with **TensorFlow Transformers** (TFAutoModelForTokenClassification)  
- Dynamic padding with a **DataCollator**  
- Monitoring metrics (accuracy, precision, recall, F1) via **seqeval** and **KerasMetricCallback**  
- Pushing the trained model to the Hugging Face Hub  
- Inference via the 🤗 `pipeline("token-classification")`  

---

## 📂 Repository Structure

```

.
├── arabic-part-of-speech.ipynb   ← Jupyter notebook with the full pipeline
├── datasets/                     ← (Optional) sample `.conllu` files
└── README.md                     ← This file

````



## ⚙️ Dependencies

* Python 3.8+
* `transformers`
* `datasets` (Hugging Face Datasets)
* `pyconll`
* `tensorflow` (2.x)
* `seqeval` (for evaluation)
* `scikit-learn` (for train\_test\_split)
* `pandas`, `numpy`

Install via:

```bash
pip install transformers datasets pyconll tensorflow seqeval scikit-learn pandas numpy
```

---

## 🛠️ Pipeline Overview

1. **Data Loading**

   * Parse a CoNLL-U file with `pyconll`
   * Extract `tokens` & `upos` tags

2. **Train/Val/Test Split**

   * 80% train, 20% temp → 70/30 split of temp

3. **Label Mapping**

   * Build `label2id` / `id2label` from the training set

4. **Tokenization & Label Alignment**

   * `AutoTokenizer.from_pretrained("asafaya/bert-base-arabic")` with `is_split_into_words=True`
   * Map word-level labels to subword tokens (`-100` for padding / special tokens)

5. **Dataset Preparation**

   * Convert to `datasets.DatasetDict`
   * Use `map(tokenize_and_align_labels)`

6. **Data Collation**

   * `DataCollatorForTokenClassification` for dynamic padding

7. **Model Setup**

   * `TFAutoModelForTokenClassification.from_pretrained(..., num_labels=17, id2label, label2id)`
   * Optimizer & LR schedule via `create_optimizer`

8. **Training**

   * `model.compile(optimizer=optimizer)`
   * Keras callbacks:

     * `KerasMetricCallback` (seqeval metrics)
     * `PushToHubCallback`

9. **Inference**

   * Use `pipeline("token-classification", model=...)`
   * Or manually tokenize → `model.predict()` → decode via `id2label`

10. **Evaluation**

    * Aggregate predictions & labels
    * Compute precision, recall, F1 with `seqeval.classification_report`

---

## 📈 Sample Results

| Metric               | Score |
| -------------------- | ----: |
| Token-level Accuracy |  0.96 |
| Macro-F1             |  0.93 |
| Precision            |  0.95 |
| Recall               |  0.92 |

*Per-tag breakdown and confusion heatmap available in the notebook.*

---

## 🤝 Contributing

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m "Add feature"`)
4. Push to branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---
