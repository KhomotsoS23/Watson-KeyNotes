import streamlit as st
import os
import asyncio
from ibm_watson import SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from dotenv import load_dotenv
# Load environment variables from .env file
# Load environment variables from .env file
load_dotenv()
# Configure IBM Watson Speech to Text
def configure_speech_to_text():
    authenticator = IAMAuthenticator(os.getenv('SPEECH_TO_TEXT_API_KEY'))
    speech_to_text = SpeechToTextV1(authenticator=authenticator)
    speech_to_text.set_service_url('https://api.eu-de.speech-to-text.watson.cloud.ibm.com/instances/39067507-f327-432f-bd98-f11e18f7c539')
    return speech_to_text
# Configure WatsonX model for text generation
def configure_watsonx():
    credentials = {
        "url": os.getenv("https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-2", "https://us-south.ml.cloud.ibm.com"),
        "apikey": os.getenv("WATSONX_API_KEY")
    }
    project_id = os.getenv("PROJECT_ID")
    model_id = "ibm/granite-13b-chat-v2"
    parameters = {
        GenParams.DECODING_METHOD: "greedy",
        GenParams.MIN_NEW_TOKENS: 100,
        GenParams.MAX_NEW_TOKENS: 2000,
        GenParams.TOP_P: 0.9,
        GenParams.TEMPERATURE: 0.7
    }
    model = ModelInference(
        model_id=model_id,
        params=parameters,
        credentials=credentials,
        project_id=project_id
    )
    return model
# Process transcript with speaker identification
def process_transcript_with_speakers(result):
    transcript = []
    current_speaker = None
    for utterance in result['results']:
        if 'speaker_labels' in utterance:
            speaker = utterance['speaker_labels'][0]['speaker']
            if speaker != current_speaker:
                current_speaker = speaker
                transcript.append(f"\nSpeaker {speaker}:")
        if 'alternatives' in utterance and len(utterance['alternatives']) > 0:
            transcript.append(utterance['alternatives'][0]['transcript'])
    return ' '.join(transcript)
# Summarize transcript with WatsonX model with structured output
async def generate_structured_summary_async(model, transcript):
    prompt = (
        "Analyze the transcript below and provide a structured meeting summary with the following sections:\n\n"
        "1. **Context**: Provide a brief description of the purpose and participants of the meeting.\n"
        "2. **Key Points**: Summarize the main topics and discussion points covered during the meeting.\n"
        "3. **Action Items**: List any tasks assigned to participants.\n"
        "4. **Next Steps**: Describe planned follow-ups or meetings.\n"
        "5. **Conclusion**: Summarize the overall status and outcome of the meeting.\n\n"
        "Transcript:\n\n"
        f"{transcript}"
    )
    response = await asyncio.to_thread(model.generate, prompt)
    return response.get("results")[0]["generated_text"]
# Streamlit app setup with structured summary for manual transcript
def main():
    st.title("Watson KeyNotes")
    # Toggle for speaker identification
    identify_speakers = st.checkbox("Identify Speakers", value=True)
    # Options for input: Upload audio file or enter transcript
    st.subheader("Input Options")
    option = st.selectbox("Choose input type:", ["Audio File", "Manual Transcript"])
    if option == "Audio File":
        uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac"])
    else:
        uploaded_transcript = st.text_area("Enter transcript manually", height=150)
    if st.button("Generate Meeting Notes"):
        if option == "Audio File" and uploaded_file is not None:
            st.info("Transcribing the audio...")
            # Initialize the Speech to Text service
            speech_to_text = configure_speech_to_text()
            try:
                result = speech_to_text.recognize(
                    audio=uploaded_file,
                    content_type=uploaded_file.type,
                    model='en-US_BroadbandModel',
                    speaker_labels=identify_speakers
                ).get_result()
                # Process transcript with or without speaker identification
                if identify_speakers:
                    transcript = process_transcript_with_speakers(result)
                else:
                    transcript = ' '.join([r['alternatives'][0]['transcript'] for r in result['results']])
                st.success("Transcription complete!")
                st.text_area("Transcript", value=transcript, height=300)
                # Generate the structured summary
                st.info("Generating structured summary...")
                model = configure_watsonx()
                call_summary = asyncio.run(generate_structured_summary_async(model, transcript))
                st.success("Structured Summary generated!")
                st.text_area("Meeting Summary", value=call_summary, height=300)
            except Exception as e:
                st.error(f"Error during transcription or summarization: {str(e)}")
        elif option == "Manual Transcript" and uploaded_transcript:
            st.info("Generating structured summary...")
            try:
                # Initialize WatsonX model for text summarization
                model = configure_watsonx()
                call_summary = asyncio.run(generate_structured_summary_async(model, uploaded_transcript))
                st.success("Structured Summary generated!")
                st.text_area("Meeting Summary", value=call_summary, height=300)
            except Exception as e:
                st.error(f"Error during summarization: {str(e)}")
        else:
            st.warning("Please upload an audio file or enter a transcript to proceed.")
if __name__ == "__main__":
    main()







