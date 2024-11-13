import os
import logging
import asyncio
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from ibm_watson import SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
# Load environment variables
load_dotenv()
# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
app = Flask(__name__)
# Configure IBM Watson Speech to Text
def configure_speech_to_text():
    authenticator = IAMAuthenticator(os.getenv("SPEECH_TO_TEXT_API_KEY"))
    speech_to_text = SpeechToTextV1(authenticator=authenticator)
    speech_to_text.set_service_url(os.getenv("SPEECH_TO_TEXT_URL"))
    return speech_to_text
# Configure WatsonX model for summarization
def configure_watsonx():
    credentials = {
        "url": os.getenv("WATSONX_URL"),
        "apikey": os.getenv("WATSONX_API_KEY")
    }
    project_id = os.getenv("PROJECT_ID")
    model_id = "mistralai/mistral-large"
    parameters = {
        GenParams.DECODING_METHOD: "greedy",
        GenParams.MIN_NEW_TOKENS: 100,
        GenParams.MAX_NEW_TOKENS: 2000,
        GenParams.TOP_P: 0.9,
        GenParams.TEMPERATURE: 0.7
    }
    return ModelInference(model_id=model_id, params=parameters, credentials=credentials, project_id=project_id)
# Endpoint for generating meeting notes from transcript or audio file
@app.route("/generateMeetingNotes", methods=["POST"])
def generate_meeting_notes():
    transcript = request.form.get("transcript")
    audio_file = request.files.get("audio")
    if not transcript and not audio_file:
        return jsonify({"error": "Please provide either a transcript or an audio file"}), 400
    # Transcribe audio if provided
    if audio_file:
        speech_to_text = configure_speech_to_text()
        try:
            result = speech_to_text.recognize(
                audio=audio_file,
                content_type=audio_file.content_type
            ).get_result()
            transcript = ' '.join([r["alternatives"][0]["transcript"] for r in result["results"]])
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return jsonify({"error": "Audio transcription failed"}), 500
    # Generate summary
    model = configure_watsonx()
    prompt = f"Summarize the following meeting transcript:\n{transcript}"
    try:
        response = asyncio.run(model.generate(prompt))
        meeting_summary = response.get("results")[0]["generated_text"]
        return jsonify({"meeting_summary": meeting_summary}), 201
    except Exception as e:
        logger.error(f"Error generating meeting notes: {e}")
        return jsonify({"error": "Meeting notes generation failed"}), 500
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)









