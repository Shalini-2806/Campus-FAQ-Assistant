# Campus FAQ Assistant

A local, customizable FAQ chatbot built with Streamlit. It ranks a question
against FAQ questions and their alternate phrasings using NLTK stemming and a
weighted combination of word- and character-level TF-IDF cosine similarity.
No model download, API key, or network connection is needed at runtime.

## Run on Windows

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit in your browser.

## Customize the FAQs

Edit [`faqs.csv`](faqs.csv), or upload a CSV in the app sidebar. The CSV must
include:

| Column | Required | Description |
| --- | --- | --- |
| `question` | Yes | The canonical FAQ question |
| `answer` | Yes | The answer displayed to the user |
| `alternate_questions` | No | Other ways to ask the same question, separated by `\|` |

Questions and answers cannot be blank, and canonical questions must be unique.
The app validates uploaded files and shows a useful error when a file is invalid.
Use **Download the FAQ template** to get the included CSV.

The included course catalog lists common engineering, arts, and arts-and-science
programs as examples. It is not a complete or official list for every institution.
Replace the sample answers with verified details from the college you represent.

## Matching and privacy

Text is normalized and stemmed locally. Word unigrams/bigrams and character
n-grams are compared against each FAQ and its alternate phrasings. A configurable
confidence threshold prevents low-similarity questions from being presented as
answers. Chat history stays in the current Streamlit session; no third-party
service receives questions.

## Run tests

```powershell
python -m unittest discover -s tests -v
```
