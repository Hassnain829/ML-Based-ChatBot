# BMYBrand Chatbot

A Flask chatbot for answering common questions about BMYBrand services. It classifies each message into a predefined intent and returns one of the answers written for that intent. The current dataset covers services, package pricing, design and website features, contact information, and business policies.

The bot uses a local **TF-IDF + logistic regression** model. Its answers come from the project data; it does not generate new text or look up live business information.

## Features

- Browser chat interface with a typing indicator and Enter-to-send support.
- 37 intents, 1,364 example questions, and 80 prepared responses in the current dataset.
- JSON chat endpoint and a model health endpoint.
- Training script that rebuilds the model after the intent data changes.
- Bundled trained model files, so training is optional for the initial run.

## Covered topics

- Service and package questions about logos, websites, e-commerce, branding, video, SEO, social media, and content.
- Business questions about refunds, privacy, contact options, payments, turnaround times, ownership, revisions, discounts, and support.
- Basic greetings and goodbyes.

## Project structure

```text
app.py                    Flask routes and development server
chatbot/
  bot.py                  Loads the model and selects replies
  trainer.py              Trains and saves the classifier
data/
  intents.json            Intent tags, example questions, and answers
models/
  classifier.pkl          Trained TF-IDF/classifier pipeline
  labels.pkl              Saved intent labels
  response_dict.pkl       Saved answers by intent
templates/
  index.html              Chat page and browser-side JavaScript
static/
  style.css               Chat page styling
```

## Run locally

The project was checked with **Python 3.14.3**, Flask 3.1.3, NLTK 3.9.3, NumPy 2.4.2, and scikit-learn 1.8.0. The package versions below match the environment used to load the bundled model files.

From the project root, create a virtual environment:

```sh
python -m venv .venv
```

Activate it in **Windows PowerShell**:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or activate it on **macOS/Linux**:

```sh
source .venv/bin/activate
```

Install the required packages and NLTK data:

```sh
python -m pip install "Flask==3.1.3" "nltk==3.9.3" "numpy==2.4.2" "scikit-learn==1.8.0"
python -m nltk.downloader punkt punkt_tab wordnet
```

Start the Flask development server:

```sh
python -X utf8 app.py
```

Open **http://localhost:5000**. The `-X utf8` flag also avoids console encoding errors from the app's Unicode log messages on some Windows terminals.

The included `models/*.pkl` files are loaded when the app starts. If they cannot be loaded with your Python or scikit-learn version, run the training command below to create compatible files.

## How it works

1. The browser sends a message to `POST /chat`.
2. The bot lowercases, tokenizes, and lemmatizes the message with NLTK.
3. A TF-IDF vectorizer using one-, two-, and three-word phrases feeds a logistic regression intent classifier.
4. The bot selects one of the prepared responses for the predicted intent. Multiple runs can return different wording for the same intent.

Each message is classified independently; the app does not store a conversation history.

The training script reads `data/intents.json`, fits the pipeline, prints an optional five-fold cross-validation result, and writes the three files in `models/`.

## API

| Route | Method | Purpose |
| --- | --- | --- |
| `/` | `GET` | Serve the chat interface. |
| `/chat` | `POST` | Accept JSON such as `{"message":"What services do you offer?"}` and return `{"response":"..."}`. A blank message returns HTTP 400 with `{"error":"No message provided"}`. |
| `/health` | `GET` | Return `{"status":"ok","model_loaded":true}` when the model loaded successfully. Check `model_loaded`; `status` is always `"ok"` in the current implementation. |

For example, with the server running:

```sh
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What services do you offer?"}'
```

In Windows PowerShell, you can use:

```powershell
Invoke-RestMethod -Uri http://localhost:5000/chat -Method Post -ContentType 'application/json' -Body '{"message":"What services do you offer?"}'
```

## Update answers or train the model

Edit `data/intents.json`. Each entry has a `tag`, a list of example `patterns`, and a list of `responses`. Then run:

```sh
python -X utf8 -m chatbot.trainer
```

The trainer overwrites the files in `models/`. Restart the Flask app afterward because it loads those files only at startup. The trainer downloads the required NLTK data if it is missing.

You can also try the command-line chat:

```sh
python -X utf8 -m chatbot.bot
```

## Before public deployment

This repository currently runs Flask with debug mode enabled and inserts chat text into the page as HTML. Address those behaviors and use a production server before exposing the app publicly. The bot also prints messages and prediction details to server logs.

Prices, policies, contact details, and privacy statements are stored as fixed responses in `data/intents.json`. Review them with the business owner before publishing; editing the JSON requires retraining. The classifier can assign an unfamiliar question to an existing intent, so review its responses before using it for customer-facing decisions. Only load model pickle files from a source you trust.
