def main():
    print(
        "\n",
        "The Proof-of-Concept (PoC) is a Minimum Viable Product (MVP) pipeline designed to automate the classification of free-text feedback comments into positive or negative sentiment",
    )


# ----------------------------------------------------------------------
# 1. Setup and Imports
# ----------------------------------------------------------------------
# Project Dependencies (to be managed by uv):
# uv add sieves transformers

from sieves import Pipeline, tasks, Doc
from transformers import pipeline

# ----------------------------------------------------------------------
# 2. Define Inputs (Simulated TEP User Feedback)
# ----------------------------------------------------------------------

# TEP collects feedback from pupils and employees.
# Simulated comments based on educational feedback datasets:
# Revised comments reflecting language and common issues/praise in UK secondary schools:
simulated_comments = [
    # Positive feedback (Focus on learning environment/teacher skill)
    Doc(text="My History lessons are really clear and the teacher gives good support."),
    # Negative/Improvement suggestion (Focus on pacing/classroom environment)
    Doc(text="The teacher goes too fast and the seating plan is rubbish."),
    # Mixed sentiment, categorized here as negative/actionable (Highlighting workload issue)
    Doc(
        text="The Science course is great, but we get way too much homework every night."
    ),
    # Highly positive comment (Praise for staff member)
    Doc(
        text="Miss Smith is the best! She always makes sure everyone understands the topic."
    ),
    # Actionable Negative (Suggests operational or resource deficiency)
    Doc(text="The lunch queue is too long, it wastes all of break time."),
    # Specific negative emotional feedback (Issue related to behaviour/peers/subject)
    Doc(text="It's hard to focus in English because of the noise on the back tables."),
]

# Define the target classification labels.
CLASSIFICATION_LABELS = ["positive", "negative"]

# ----------------------------------------------------------------------
# 3. Model and Pipeline Configuration
# ----------------------------------------------------------------------

# We use the Hugging Face zero-shot classification pipeline, which eliminates
# the need for TEP to manage MLOps infrastructure (Hugging Face Inference
# Endpoints are a managed solution highly suitable for TEP's capacity).
classification_model = pipeline(
    "zero-shot-classification",
    # Using a suitable, lightweight zero-shot model
    model="MoritzLaurer/xtremedistil-l6-h256-zeroshot-v1.1-all-33",
)

# The sieves Pipeline provides the document-based architecture
# for observability and guaranteed structured output.
sentiment_pipeline = Pipeline(
    [
        tasks.Classification(
            labels=CLASSIFICATION_LABELS,  # The defined output structure
            model=classification_model,  # The Hugging Face model wrapper
            task_id="sentiment_analysis",  # Tags the result for later extraction
        )
    ]
)

# ----------------------------------------------------------------------
# 4. Run Pipeline and Output Results
# ----------------------------------------------------------------------

print(
    f"--- Running TEP Sentiment Analysis MVP Pipeline ({len(simulated_comments)} Documents) ---"
)
# The pipeline takes Doc objects and outputs processed Doc objects.
classified_docs = list(sentiment_pipeline(simulated_comments))

# Output the results, demonstrating the structured and trackable output
for idx, doc in enumerate(classified_docs):
    text = doc.text
    results = doc.results.get("sentiment_analysis", [])

    # Extract the top label and score if results are valid
    # Results are a list of tuples: [(label, score), ...]
    if results and isinstance(results, list) and len(results) > 0:
        # Get the first tuple (highest confidence prediction)
        best_prediction, confidence = results[0]

        # Display the input text and the structured output prediction
        print(f"\n[Comment {idx + 1}]")
        print(f"Text: {text[:70]}...")
        print(
            f"-> CLASSIFICATION: {best_prediction.upper()} (Confidence: {confidence:.2f})"
        )
    else:
        print(f"\n[Comment {idx + 1}] Could not classify or structure was unexpected.")


if __name__ == "__main__":
    main()
