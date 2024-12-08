
# NLP Project: Analyzing Shifts in Reddit User Opinions 🌍📊

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)](https://www.python.org/)
[![NLP](https://img.shields.io/badge/NLP-Transformers-orange)](https://huggingface.co/)
[![Contributors](https://img.shields.io/badge/Contributors-5-purple)](#authors)

This project investigates how user opinions and sentiments on Reddit shift **before and after significant events**, particularly focusing on the **Israel-Palestine conflict following October 7, 2023**. By applying cutting-edge **NLP techniques** and **social network analysis**, this project uncovers the dynamics of public discourse during global events.

---

## 📖 Introduction

Reddit provides a unique lens into societal discourse, especially during critical global events. This project aims to:

- **Explore sentiment shifts** among Reddit users in response to the Israel-Palestine conflict.
- **Analyze data trends** from millions of posts and comments across relevant subreddits.
- **Bridge the gap** between computational analysis and social understanding, offering insights into how opinions evolve in digital spaces during conflicts.

---

## 📊 Data

Two key datasets are used in this analysis:

1. **Kaggle Dataset**:
   - Tabular dataset with over **2 million posts** from the `IsraelPalestine` and `worldnews` subreddits.
   - Captures **high-level trends** in sentiment and user activity.

2. **Raw Reddit Data**:
   - JSON-format dataset including posts and comments from the `IsraelPalestine` subreddit since **January 2023**.
   - Provides **tree-structured data** for detailed conversational dynamics.

---

## 🔬 Methodology

### 1. **Data Collection and Preparation**
   - **Preprocessing**: Removed irrelevant metadata for consistency and quality.
   - **Labeling**: Classified stances as **Pro-Israel**, **Pro-Palestine**, or **Neutral** using a mix of automated and manual methods.

### 2. **Fine-Tuning Language Models**
   - Leveraged models like **BERT** and **RoBERTa** for sentiment classification.
   - Explored **zero-shot** and **few-shot learning** using models like **GPT** and **Llama** for nuanced stances.

### 3. **Dynamic Social Network Analysis**
   - Built interaction networks from Reddit’s tree-structured data.
   - Mapped sentiment and stance changes across **user interactions** over time.

### 4. **Event Correlation & Timeline Analysis**
   - Cross-referenced **key global events** with user sentiment trends.
   - Analyzed **individual user timelines** to identify turning points in stances.

---

## 🔍 Key Findings

- Significant shifts in sentiment before and after major events.
- Identification of critical dates correlating with **peaks in discourse activity**.
- Insights into how **polarized discussions propagate** in digital communities.

---

## 📂 Project Structure

```plaintext
├── data/
│   ├── evaluation/
│   ├── experiments/
│   ├── notebooks/
│   ├── papers/
│   └── social_network/
├── scripts/
├── README.md
