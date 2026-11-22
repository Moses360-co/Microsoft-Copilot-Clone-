from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Supported languages
LANGUAGE_CODES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Chinese": "zh",
    "Japanese": "ja",
    "Korean": "ko",
    "Russian": "ru",
    "Arabic": "ar",
    "Tamil": "ta"
}

# Customer support responses
SUPPORT_RESPONSES = {
    "greeting": (
        "Hello! Welcome to our customer support. "
        "How can I help you today?"
    ),
    "product": (
        "Our product comes with a 30-day warranty. "
        "Which product would you like to know about?"
    ),
    "pricing": (
        "Our pricing starts at $9.99 per month. "
        "Would you like to know more about our plans?"
    ),
    "warranty": (
        "Our product comes with a 30-day warranty. "
        "Please keep your purchase receipt for warranty claims."
    ),
    "contact": (
        "You can contact our support team at "
        "support@example.com or call 1-800-123-4567."
    ),
    "refund": (
        "For refund assistance, please contact our support team "
        "with your order details."
    ),
    "shipping": (
        "Shipping times depend on your location. "
        "Please provide your order details for more information."
    ),
    "thanks": (
        "You're welcome! Please let us know if you need further help."
    ),
    "goodbye": (
        "Thank you for contacting us. Have a great day!"
    ),
    "default": (
        "I'm sorry, I couldn't understand your question. "
        "You can ask about products, pricing, warranty, refunds, "
        "shipping, or contact information."
    )
}

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Load the translation model
MODEL_NAME = "facebook/m2m100_418M"

try:
    print("Loading translation model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    model.to(DEVICE)
    model.eval()

    print("Translation model loaded successfully!")

except Exception:
    app.logger.exception("Failed to load translation model")
    tokenizer = None
    model = None


def translate_text(text, source_language, target_language):
    """Translate text using the M2M100 model."""

    if not isinstance(text, str) or not text.strip():
        return ""

    if source_language not in LANGUAGE_CODES:
        raise ValueError(f"Unsupported source language: {source_language}")

    if target_language not in LANGUAGE_CODES:
        raise ValueError(f"Unsupported target language: {target_language}")

    if source_language == target_language:
        return text

    if tokenizer is None or model is None:
        raise RuntimeError(
            "Translation model is not loaded. Check the console for errors."
        )

    source_code = LANGUAGE_CODES[source_language]
    target_code = LANGUAGE_CODES[target_language]

    # Set the source language explicitly.
    tokenizer.src_lang = source_code

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

    # Force generation in the selected target language.
    target_token_id = tokenizer.get_lang_id(target_code)

    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=target_token_id,
            max_new_tokens=256,
            num_beams=4
        )

    translated_text = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True
    )[0]

    return translated_text.strip()


def get_support_response(message):
    """Identify the customer's intent from their English message."""

    message = message.lower().strip()

    # Check specific intents before general intents.
    intents = {
        "greeting": [
            "hello", "hi", "hey", "good morning",
            "good afternoon", "good evening"
        ],
        "pricing": [
            "price", "pricing", "cost", "how much",
            "payment", "fee", "subscription", "expensive"
        ],
        "warranty": [
            "warranty", "guarantee", "repair", "damaged"
        ],
        "refund": [
            "refund", "money back", "return my money",
            "return a product", "cancel my order"
        ],
        "shipping": [
            "shipping", "delivery", "deliver", "tracking",
            "order status", "where is my order"
        ],
        "contact": [
            "contact", "support", "phone", "email",
            "customer service", "talk to someone"
        ],
        "thanks": [
            "thank you", "thanks", "appreciate it"
        ],
        "goodbye": [
            "goodbye", "bye", "see you"
        ],
        "product": [
            "product", "products", "item", "items",
            "feature", "features"
        ]
    }

    for intent, keywords in intents.items():
        if any(keyword in message for keyword in keywords):
            return SUPPORT_RESPONSES[intent]

    return SUPPORT_RESPONSES["default"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json(silent=True) or {}

    user_message = data.get("message", "")
    user_language = data.get("language", "English")

    if not isinstance(user_message, str) or not user_message.strip():
        return jsonify({
            "error": "Please enter a message."
        }), 400

    if user_language not in LANGUAGE_CODES:
        return jsonify({
            "error": "Unsupported language selected."
        }), 400

    if tokenizer is None or model is None:
        return jsonify({
            "error": "Translation model failed to load. Check the PyCharm terminal."
        }), 503

    try:
        # STEP 1: Translate the customer's message into English.
        if user_language == "English":
            english_message = user_message.strip()
        else:
            english_message = translate_text(
                user_message,
                user_language,
                "English"
            )

        app.logger.info("Customer message received and translated.")

        # STEP 2: Identify the customer's question.
        english_response = get_support_response(english_message)

        # STEP 3: Translate the response into the selected language.
        if user_language == "English":
            final_response = english_response
        else:
            final_response = translate_text(
                english_response,
                "English",
                user_language
            )

        # STEP 4: Send the result to the frontend.
        return jsonify({
            "response": final_response,
            "detected_message": english_message
        })

    except Exception:
        app.logger.exception("Chat processing failed")

        return jsonify({
            "error": (
                "Translation failed. Please check the PyCharm terminal "
                "for the exact error."
            )
        }), 500


if __name__ == "__main__":
    app.run(debug=True)