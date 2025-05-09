import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import os
from datetime import datetime
import speech_recognition as sr  # For audio upload (fallback)
import pyttsx3
from io import BytesIO
from pdfminer.high_level import extract_text as pdf_extract_text  # For PDF extraction
from docx import Document as DocxDocument  # For DOCX extraction

# MUST BE THE VERY FIRST STREAMLIT COMMAND
st.set_page_config(page_title="DoctorBot", layout="centered")

# Initialize pyttsx3 engine (using st.cache_resource for Streamlit compatibility)
@st.cache_resource
def get_tts_engine():
    engine = pyttsx3.init()
    return engine

engine = get_tts_engine()

def speak(text):
    if st.session_state.get("speak_responses", True):
        engine.say(text)
        try:
            engine.runAndWait()
        except RuntimeError as e:
            if "run loop already started" not in str(e):
                raise

# ===== FEATURE FLAGS CONFIGURATION =====
if 'feature_flags' not in st.session_state:
    st.session_state.feature_flags = {
        "use_ai_model": True,
        "use_expanded_faq": True,
        "use_cache": True,
        "logging_enabled": True,
        "debug_mode": True,
        "speak_responses": True,
        "use_realtime_speech": False,  # Placeholder for enabling custom component
    }

# ===== MODEL LOADING =====
@st.cache_resource
def load_qa_model():
    try:
        model_name = "dmis-lab/biobert-base-cased-v1.1"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        return {"tokenizer": tokenizer, "model": model, "loaded": True}
    except Exception as e:
        if st.session_state.feature_flags["debug_mode"]:
            st.error(f"Model Loading Error: {str(e)}")
        return {"loaded": False, "error": str(e)}

# ===== QUESTION ANSWERING =====
def get_model_answer(question, context):
    if not st.session_state.feature_flags["use_ai_model"]:
        return None
    model_data = load_qa_model()
    if not model_data["loaded"]:
        return None
    try:
        tokenizer = model_data["tokenizer"]
        model = model_data["model"]
        inputs = tokenizer.encode_plus(question, context, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        start = torch.argmax(outputs.start_logits)
        end = torch.argmax(outputs.end_logits) + 1
        if end > start:
            answer_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start:end])
            answer = tokenizer.convert_tokens_to_string(answer_tokens).strip()
            return answer
        return None
    except Exception as e:
        if st.session_state.feature_flags["debug_mode"]:
            st.error(f"QA Error: {str(e)}")
        return None

# ===== CACHING =====
if 'response_cache' not in st.session_state:
    st.session_state.response_cache = {}

def get_cached_response(question):
    if st.session_state.feature_flags["use_cache"] and question in st.session_state.response_cache:
        return st.session_state.response_cache[question]
    return None

def cache_response(question, response):
    if st.session_state.feature_flags["use_cache"]:
        st.session_state.response_cache[question] = response

# ===== FAQ DATABASE =====
basic_faq_responses = {
    "who are you": "Hello, I am DoctorBot, your virtual health assistant.",
    "what can you do": "I can answer your health-related questions, explain symptoms, and offer basic guidance. I can also analyze medical documents.",
    "how are you": "I'm doing well! How can I help you today?",
    "what is malaria": "Malaria is a disease caused by a parasite transmitted by mosquitoes. Symptoms include fever, chills, and sweating.",
    "what is anemia": "Anemia is when your body lacks enough healthy red blood cells to carry oxygen.",
    "what is fever": "Fever is a rise in body temperature, usually due to infection or illness.",
    "what are chronic diseases": "Chronic diseases are long-term illnesses like diabetes, heart disease, and hypertension."
}

extended_faq_responses = {
    "what is diabetes": "Diabetes affects how your body processes blood sugar. It requires management via diet, exercise, or medication.",
    "how can i prevent malaria": "Use mosquito nets, avoid standing water, and use repellents.",
    "how can i prevent anemia": "Eat iron-rich foods like spinach and lentils, and take supplements if prescribed.",
    "what is the cause of fever": "Fever is often caused by infection or inflammation.",
    "what are the symptoms of dehydration": "Dry mouth, fatigue, dizziness, dark urine.",
    "what is vitamin a good for": "Good vision, immune function, and skin health.",
    "what does vitamin b do": "Supports energy, brain function, and red blood cell production.",
    "what is vitamin c important for": "Boosts immunity, helps heal wounds, and improves iron absorption.",
    "benefits of vitamin d": "Helps calcium absorption and boosts immunity.",
    "how do i get enough vitamin e": "Eat nuts, seeds, and leafy greens.",
    "why is vitamin k important": "Important for blood clotting and bone health.",
    "role of water": "Water supports all bodily functions and keeps you hydrated.",
    "how much water should i drink per day": "2–3 liters daily depending on activity and climate.",
    "foods rich in iron": "Spinach, beans, red meat, tofu, lentils.",
    "what is a balanced diet": "A balanced diet includes carbs, protein, fats, vitamins, and minerals.",
    "what are macronutrients": "Carbs, proteins, and fats — the main components of diet.",
    "role of carbohydrates": "Provide quick energy for daily activities.",
    "why should i eat proteins": "To build and repair body tissues.",
    "healthy diet plan": "Eat fruits, vegetables, lean protein, whole grains, and healthy fats.",
    "how can i stay healthy": "Eat well, exercise, sleep enough, and manage stress.",
    "how do i prevent stress": "Practice meditation, take breaks, and talk to someone.",
    "how to improve immunity": "Eat healthy, stay active, and get good sleep.",
    "foods for better skin": "Nuts, seeds, berries, and Omega-3 rich foods.",
    "suggest workout": "Try 30 minutes of daily walking or yoga.",
    "dehydration fatigue": "Yes, it can make you feel tired or dizzy.",
    "fatigue": "May be due to low sleep, iron, or stress.",
    "dizzy": "Could be dehydration or low sugar.",
    "pale and weak": "Possibly anemia — get iron checked.",
    "sore throat and fever": "Might be flu or throat infection.",
    "headache and fever": "Could be infection like malaria or viral fever.",
    "nausea": "Try ginger tea, rest, and avoid oily food.",
    "cough": "Stay hydrated, try warm fluids, and rest."
}

def get_faq_answer(question):
    if question in basic_faq_responses:
        return basic_faq_responses[question]
    if st.session_state.feature_flags["use_expanded_faq"] and question in extended_faq_responses:
        return extended_faq_responses[question]
    return None

health_context = """
Vitamins are essential micronutrients that the body needs in small amounts for various functions:
- Vitamin A supports vision, immune function, and reproduction.
- B-complex vitamins (B1 to B12) help in energy production, brain function, and red blood cell formation.
- Vitamin C is important for immunity, skin health, and iron absorption.
- Vitamin D helps in calcium absorption and bone health.
- Vitamin E is an antioxidant that protects cells.
- Vitamin K is essential for blood clotting.

Nutrients include macronutrients (carbs, proteins, fats) and micronutrients (vitamins, minerals).
They support growth, metabolism, and overall health.

Water is essential for life. It regulates body temperature, removes waste, lubricates joints, and transports nutrients.
Adults should drink at least 2 liters or 8 glasses of water per day.

Anemia:
- Anemia is a condition where you lack enough healthy red blood cells to carry adequate oxygen to your body's tissues.
- It can be caused by iron deficiency, vitamin B12 deficiency, or chronic disease.
- Prevention includes eating iron-rich foods (like leafy greens, legumes, and red meat) and getting enough vitamin B12.

Malaria:
- Malaria is a mosquito-borne infectious disease caused by a parasite.
- Prevention includes using insect repellent, sleeping under mosquito nets, and taking antimalarial medications when traveling to endemic areas.
- Treatment includes antimalarial drugs like chloroquine and artemisinin-based therapies.

Fever:
- Fever is usually a symptom of infection or inflammation.
- It can be caused by bacterial or viral infections, like flu or COVID-19.
- Prevention involves good hygiene practices, vaccination, and avoiding exposure to infected individuals.

Chronic Diseases:
- Chronic diseases include diabetes, hypertension, cardiovascular diseases, and kidney disease.
- Prevention involves a healthy diet, regular physical activity, managing stress, avoiding smoking, and maintaining a healthy weight.
"""

# ===== LOGGING =====
def log_interaction(question, answer, source):
    if st.session_state.feature_flags["logging_enabled"]:
        try:
            log_dir = "logs"
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"healthbot_log_{datetime.now().strftime('%Y%m%d')}.txt")
            with open(log_file, "a") as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{timestamp}] Q: {question}\n")
                f.write(f"[{timestamp}] A: {answer} (Source: {source})\n\n")
        except Exception as e:
            if st.session_state.feature_flags["debug_mode"]:
                st.error(f"Logging Error: {str(e)}")

# ===== AUDIO TRANSCRIPTION (for uploaded audio - fallback) =====
def transcribe_audio(audio_file):
    r = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            try:
                audio = r.record(source)
                text = r.recognize_google(audio)  # Still uses an API
                return text
            except sr.UnknownValueError:
                return "Could not understand audio"
            except sr.RequestError as e:
                return f"Could not request results from service; {e}"
    except Exception as e:
        return f"Error processing audio file: {e}"

# ===== DOCUMENT ANALYSIS (BASIC PLACEHOLDER) =====
def analyze_medical_document(file):
    text = ""
    if file.type == "application/pdf":
        try:
            text = pdf_extract_text(BytesIO(file.getvalue()))
        except Exception as e:
            return f"Error reading PDF: {e}"
    elif file.type == "text/plain":
        text = file.getvalue().decode("utf-8")
    elif "officedocument.wordprocessingml.document" in file.type:
        try:
            doc = DocxDocument(BytesIO(file.getvalue()))
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        except Exception as e:
            return f"Error reading DOCX: {e}"
    else:
        return "Unsupported file type."

    if text:
        # Basic analysis placeholder - you would replace this with your actual logic
        return f"Document analysis performed. Extracted text:\n\n{text[:500]}...\n\nFurther analysis would go here."
    else:
        return "No text could be extracted from the document."

# ===== MAIN APP =====
def main():
    st.title("🩺 DoctorBot") # Moved after set_page_config, though order doesn't strictly matter for content
    if 'introduction_spoken' not in st.session_state:
        introduction_text = "Hello, I am DoctorBot, your virtual health assistant. You can ask me questions by typing, speaking (using the microphone option), or by uploading a medical document for analysis."
        speak(introduction_text)
        st.session_state['introduction_spoken'] = True

    with st.sidebar:
        st.header("Feature Flags")
        for flag, value in st.session_state.feature_flags.items():
            new_value = st.checkbox(
                flag.replace("_", " ").title(),
                value=value,
                key=f"flag_{flag}"
            )
            st.session_state.feature_flags[flag] = new_value

        if st.session_state.feature_flags["debug_mode"]:
            st.header("System Status")
            if st.session_state.feature_flags["use_ai_model"]:
                model_data = load_qa_model()
                if model_data["loaded"]:
                    st.success("BioBERT Model: Loaded")
                else:
                    error_msg = model_data.get("error", "Not initialized")
                    st.error(f"BioBERT Model: Failed to load\n{error_msg}")
            else:
                st.info("BioBERT Model: Disabled")
            if st.session_state.feature_flags["use_cache"]:
                st.success(f"Cache: Enabled ({len(st.session_state.response_cache)} entries)")
            else:
                st.info("Cache: Disabled")
            if st.button("Clear Cache"):
                st.session_state.response_cache = {}
                st.success("Cache cleared!")

    st.markdown("Ask DoctorBot by typing, speaking into your microphone, or uploading a medical document for analysis.")

    input_option = st.radio("How would you like to interact?", ("Type Question", "Speak Question (Microphone)", "Upload Medical Document"))

    user_question = None

    if input_option == "Type Question":
        st.markdown("**Type your health-related question below:**")
        user_question = st.text_input("Type here:")
    elif input_option == "Speak Question (Microphone)":
        st.warning("Real-time microphone input requires a custom Streamlit component. This option will be a placeholder. For now, you can use 'Upload Medical Document' for file-based input or 'Type Question'.")
        # Placeholder for the custom microphone component
        # If you had a component named 'realtime_mic_input', you would use it here:
        # user_question = realtime_mic_input()
        pass
    elif input_option == "Upload Medical Document":
        medical_file = st.file_uploader("Upload your medical document (PDF, TXT, DOCX):", type=["pdf", "txt", "docx"])
        if medical_file is not None:
            with st.spinner("Analyzing document..."):
                analysis_result = analyze_medical_document(medical_file)
            st.subheader("Document Analysis Result:")
            st.write(analysis_result)
            speak("Document analysis complete.") # Optional speech feedback
            return # Important: Don't process as a regular question

    if st.button("Ask"):
        if user_question:
            clean_question = user_question.lower().strip()
            cached_response = get_cached_response(clean_question)
            if cached_response:
                st.success(cached_response)
                speak(cached_response)
                return

            response = get_faq_answer(clean_question)
            source = "FAQ"

            if not response and st.session_state.feature_flags["use_ai_model"]:
                answer = get_model_answer(clean_question, health_context)
                if answer and len(answer) > 5:
                    response = answer
                    source = "AI Model"

            if response:
                st.success(response)
                log_interaction(clean_question, response, source)
                cache_response(clean_question, response)
                speak(response)
            else:
                default_response = "I'm sorry, I don't understand that yet. Try asking about a health topic or vitamin."
                st.warning(default_response)
                log_interaction(clean_question, default_response, "Default")
                speak(default_response)
        elif input_option == "Type Question" and not user_question:
            st.warning("Please type a question to proceed.")
        elif input_option == "Upload Medical Document" and 'medical_file' not in locals():
            st.warning("Please upload a medical document to proceed with analysis.")

if __name__ == "__main__":
    main()