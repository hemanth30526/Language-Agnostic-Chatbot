from flask import Flask, request, jsonify, render_template, Response
import os
import re
import requests
from dotenv import load_dotenv
from sarvamai import SarvamAI

load_dotenv()

app = Flask(__name__, template_folder='../frontend')

sarvam_client = SarvamAI(
    api_subscription_key=os.getenv("SARVAM_API_KEY"),
)

# Cache for parsed courses.md content
_parsed_courses_cache = None

def parse_courses_md():
    global _parsed_courses_cache
    if _parsed_courses_cache is not None:
        return _parsed_courses_cache

    courses_data = {}
    current_title = None
    current_content = []

    try:
        with open('courses.md', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if line and (i + 1 < len(lines) and lines[i+1].strip().startswith("Description:")):
                    # This line is likely a title
                    if current_title and current_content:
                        courses_data[current_title.lower()] = "\n".join(current_content).strip()
                    current_title = line
                    current_content = []
                elif current_title:
                    current_content.append(line)
                i += 1
            # Add the last course
            if current_title and current_content:
                courses_data[current_title.lower()] = "\n".join(current_content).strip()
    except FileNotFoundError:
        print("Error: courses.md not found.")
        return {}
    
    _parsed_courses_cache = courses_data
    return courses_data


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    conversation = request.json.get('conversation')
    
    # Extract the latest user message and its index
    latest_user_message_content = ""
    latest_user_message_index = -1
    for i, message in enumerate(reversed(conversation)):
        if message.get('role') == 'user' and message.get('parts') and message['parts'][0].get('text'):
            latest_user_message_content = message['parts'][0]['text']
            latest_user_message_index = len(conversation) - 1 - i # Get original index
            break

    modified_conversation = list(conversation) # Create a mutable copy

    system_instruction = ""
    response_suffix = ""
    direct_response_content = None # To store content from courses.md if found

    # --- Agent 1 (GRAPH) Logic ---
    if latest_user_message_content.strip().upper().startswith("GRAPH"):
        system_instruction = "You are Agent 1. If the user is studying or asking about a science or tech topic, generate the entire structure of the concepts and related concepts in a graph view format (e.g., using Markdown for nodes and edges, or a textual representation of a graph). Focus on providing a clear, structured overview."
        response_suffix = " hello"
        # Remove the prefix from the user's message
        modified_conversation[latest_user_message_index]['parts'][0]['text'] = latest_user_message_content[len("GRAPH"):].strip()
    
    # --- Agent 2 (COURSE) Logic ---
    elif latest_user_message_content.strip().upper().startswith("COURSE"):
        course_query_match = re.match(r"COURSE\s+(\w+)", latest_user_message_content.strip().upper())
        if course_query_match:
            course_title_query = course_query_match.group(1).lower()
            courses_data = parse_courses_md()
            
            # Find the course title in a case-insensitive way
            found_course_title = None
            for title in courses_data.keys():
                if course_title_query in title.lower(): # Check if query is part of a title
                    found_course_title = title
                    break

            if found_course_title:
                direct_response_content = courses_data[found_course_title]
                response_suffix = " bye"
            else:
                system_instruction = "You are Agent 2, a helpful assistant for course-related queries. The user asked about a course that was not found in your knowledge base. Please inform the user that the course was not found and suggest they try another course or rephrase their query."
                response_suffix = " bye"
                # Remove the prefix from the user's message
                modified_conversation[latest_user_message_index]['parts'][0]['text'] = latest_user_message_content[len("COURSE"):].strip()
        else:
            # If "COURSE" is present but no specific course is mentioned
            system_instruction = "You are Agent 2, a helpful assistant for course-related queries. The user asked about courses but did not specify a particular course. Please ask the user to specify which course they are interested in."
            response_suffix = " bye"
            modified_conversation[latest_user_message_index]['parts'][0]['text'] = latest_user_message_content[len("COURSE"):].strip()


    # If a direct response is available, return it immediately
    if direct_response_content:
        return jsonify({'response': direct_response_content + response_suffix})

    # Prepend the system instruction if one was generated
    if system_instruction:
        modified_conversation.insert(0, {"role": "user", "parts": [{"text": system_instruction}]})

    api_key = os.getenv("GEMINI_API_KEY")
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    
    headers = {
        'Content-Type': 'application/json',
        'x-goog-api-key': api_key
    }
    
    data = {
        "contents": modified_conversation # Use the modified conversation
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        response_data = response.json()
        llm_response = response_data['candidates'][0]['content']['parts'][0]['text']
        
        # Append the suffix here to guarantee it
        llm_response += response_suffix
            
        return jsonify({'response': llm_response})
    
    except requests.exceptions.RequestException as e:
        return jsonify({'error': str(e)}), 500
    except (KeyError, IndexError) as e:
        return jsonify({'error': f"Error parsing LLM response: {e}"}), 500

import tempfile

@app.route('/voice-chat', methods=['POST'])
def voice_chat():
    audio_file = request.files['audio']
    print("--- Voice Chat Request Received ---")
    print(f"Audio file: {audio_file.filename}, content type: {audio_file.content_type}")

    try:
        # Save the audio file to a temporary file with a .webm extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
            audio_file.save(temp_audio.name)
            temp_audio_path = temp_audio.name

        # Speech to Text Translation
        print("1. Calling Speech-to-Text-Translate API...")
        with open(temp_audio_path, 'rb') as f:
            stt_response = requests.post(
                "https://api.sarvam.ai/speech-to-text-translate",
                headers={
                    "api-subscription-key": os.getenv("SARVAM_API_KEY")
                },
                files={
                    'file': ('audio.webm', f, 'audio/webm')
                },
            )
        os.remove(temp_audio_path) # Clean up the temporary file
        stt_response.raise_for_status()
        stt_data = stt_response.json()
        english_text = stt_data['transcript']
        source_lang = stt_data['language_code']
        print(f"  - STT-Translate Success: {english_text} ({source_lang})")

        # LLM
        print("2. Calling Gemini API...")
        api_key = os.getenv("GEMINI_API_KEY")
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
        headers = {
            'Content-Type': 'application/json',
            'x-goog-api-key': api_key
        }
        data = {
            "contents": [{"parts": [{"text": english_text}]}]
        }
        llm_response = requests.post(url, headers=headers, json=data)
        llm_response.raise_for_status()
        llm_data = llm_response.json()
        llm_text = llm_data['candidates'][0]['content']['parts'][0]['text']
        print(f"  - Gemini Success: {llm_text}")

        return jsonify({'response': llm_text})

    except Exception as e:
        print(f"--- Voice Chat Error: {e} ---")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)