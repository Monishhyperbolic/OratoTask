import streamlit as st
import sqlite3
import re
from datetime import datetime, timedelta
from textblob import TextBlob
from deep_translator import GoogleTranslator
import joblib
import nltk
from nltk.corpus import stopwords
import calendar
from dateutil.parser import parse
import bcrypt
import logging
import os
import pandas as pd
import streamlit.components.v1 as components

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    stop_words = set(stopwords.words('english'))
except:
    stop_words = set()
    st.error("Failed to download NLTK data. Some features may not work.")

# Initialize translator
translator = GoogleTranslator()

# Load priority model and vectorizer
model_path = 'api/priority_model.joblib'
vectorizer_path = 'api/vectorizer.joblib'
priority_model = None
vectorizer = None

try:
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        priority_model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
    else:
        st.warning("Priority model or vectorizer not found. Using keyword-based prioritization.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    st.error("Failed to load priority model. Using fallback prioritization.")

# Database connection
def get_db_connection():
    db_path = 'tasks.db'
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

# Initialize database and ensure schema is up-to-date
def initialize_database():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Create users table
        cursor.execute('''CREATE TABLE IF NOT EXISTS users
                          (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT)''')
        # Create tasks table
        cursor.execute('''CREATE TABLE IF NOT EXISTS tasks
                          (id INTEGER PRIMARY KEY, user_id INTEGER, name TEXT, date TEXT, time TEXT, status TEXT, priority TEXT, category TEXT, description TEXT,
                           FOREIGN KEY(user_id) REFERENCES users(id))''')
        # Create journal table
        cursor.execute('''CREATE TABLE IF NOT EXISTS journal
                          (id INTEGER PRIMARY KEY, user_id INTEGER, content TEXT, date TEXT,
                           FOREIGN KEY(user_id) REFERENCES users(id))''')
        # Check if tasks table has description column
        cursor.execute("PRAGMA table_info(tasks)")
        columns = [col[1] for col in cursor.fetchall()]
        if 'description' not in columns:
            cursor.execute("ALTER TABLE tasks ADD COLUMN description TEXT")
            logger.info("Added description column to tasks table")
        conn.commit()

# Initialize database
initialize_database()

# Streamlit session state initialization
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'username' not in st.session_state:
    st.session_state.username = None
if 'lang' not in st.session_state:
    st.session_state.lang = 'en-US'
if 'voice_command' not in st.session_state:
    st.session_state.voice_command = ''
if 'voice_journal' not in st.session_state:
    st.session_state.voice_journal = ''
if 'voice_data' not in st.session_state:
    st.session_state.voice_data = None
if 'task_description' not in st.session_state:
    st.session_state.task_description = ''

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

def translate_command(command: str, lang: str):
    if lang != 'en-US':
        try:
            return translator.translate(command, source=lang.split('-')[0], target='en')
        except Exception as e:
            logger.warning(f"Translation failed: {e}")
            return command
    return command

def translate_response(message: str, lang: str):
    if lang != 'en-US':
        try:
            return translator.translate(message, source='en', target=lang.split('-')[0])
        except Exception as e:
            logger.warning(f"Translation failed: {e}")
            return message
    return message

def parse_date(command: str):
    try:
        match = re.search(r'(?:on\s)?(\w+\s+\d{1,2},\s*\d{4})', command, re.IGNORECASE)
        if match:
            return parse(match.group(1)).strftime('%Y-%m-%d')
        match = re.search(r'(today|tomorrow)', command, re.IGNORECASE)
        if match:
            day = match.group(1).lower()
            return datetime.now().strftime('%Y-%m-%d') if day == 'today' else (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        return datetime.now().strftime('%Y-%m-%d')
    except Exception as e:
        logger.warning(f"Date parsing failed: {e}")
        return datetime.now().strftime('%Y-%m-%d')

def format_date_for_display(date_str: str):
    try:
        task_date = datetime.strptime(date_str, '%Y-%m-%d')
        today = datetime.now().date()
        tomorrow = today + timedelta(days=1)
        if task_date.date() == today:
            return "today"
        elif task_date.date() == tomorrow:
            return "tomorrow"
        else:
            return f"on {task_date.strftime('%Y-%m-%d')}"
    except ValueError:
        return "today"

def map_label_to_priority(label):
    if label in [0]:
        return 'low'
    elif label in [1, 2]:
        return 'medium'
    elif label in [3, 4, 5]:
        return 'high'
    return 'low'

def parse_time(time_str):
    if not time_str:
        return 'anytime'
    try:
        time_obj = datetime.strptime(time_str, '%I:%M %p') if ':' in time_str else datetime.strptime(time_str, '%I %p')
        return time_obj.strftime('%I%p').lower()
    except ValueError:
        try:
            time_obj = datetime.strptime(time_str, '%H:%M')
            return time_obj.strftime('%I%p').lower()
        except ValueError:
            return 'anytime'

def format_task_for_display(task):
    date_display = format_date_for_display(task['date'])
    name = task['name'].lower()
    time = task['time']
    description = task['description'][:50] if task['description'] else 'no description'
    if time == 'anytime':
        return f"{date_display} {name}: {description}"
    else:
        return f"{date_display} {name} at {time}: {description}"

def get_sentiment(text):
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity < -0.1:
            return 'Negative'
        elif polarity > 0.1:
            return 'Positive'
        else:
            return 'Neutral'
    except Exception as e:
        logger.warning(f"Sentiment analysis failed: {e}")
        return 'Neutral'

def parse_command(command: str, user_id: int):
    if not isinstance(command, str):
        command = str(command)
    command = command.lower()
    clean_command = preprocess_text(command)

    # ML-based priority detection
    priority = 'low'
    if priority_model and vectorizer:
        try:
            X_vec = vectorizer.transform([clean_command])
            label = priority_model.predict(X_vec)[0]
            priority = map_label_to_priority(int(label))
        except Exception as e:
            logger.error(f"Priority model prediction failed: {e}")

    # Keyword-based fallback
    priority_keywords = {
        'urgent': 'high', 'important': 'high', 'asap': 'high', 'critical': 'high', 'immediately': 'high',
        'now': 'high', 'emergency': 'high', 'high priority': 'high', 'must': 'high', 'deadline': 'high',
        'due': 'high', 'assignment': 'high', 'exam': 'high', 'study': 'high', 'test': 'high',
        'submit': 'high', 'project': 'high', 'interview': 'high', 'work': 'high', 'job': 'high',
        'career': 'high', 'meeting': 'high', 'health checkup': 'high', 'medication': 'high',
        'soon': 'medium', 'today': 'medium', 'next': 'medium', 'before eod': 'medium', 'follow up': 'medium',
        'prepare': 'medium', 'plan': 'medium', 'practice': 'medium', 'revision': 'medium', 'email': 'medium',
        'call': 'medium', 'meeting prep': 'medium', 'grocery': 'medium', 'cleaning': 'medium', 'cooking': 'medium',
        'exercise': 'medium', 'walk': 'medium', 'workout': 'medium', 'laundry': 'medium',
        'whenever': 'low', 'someday': 'low', 'later': 'low', 'eventually': 'low', 'optional': 'low',
        'gaming': 'low', 'play': 'low', 'netflix': 'low', 'youtube': 'low', 'scrolling': 'low',
        'social media': 'low', 'movie': 'low', 'fun': 'low', 'hangout': 'low', 'party': 'low',
        'rest': 'low', 'sleep in': 'low', 'nap': 'low', 'binge': 'low'
    }
    for keyword, prio in priority_keywords.items():
        if keyword in command:
            priority = prio
            break

    # Sentiment adjustment
    try:
        blob = TextBlob(command)
        sentiment_score = blob.sentiment.polarity
        sentiment_label = 'negative' if sentiment_score < -0.1 else 'positive' if sentiment_score > 0.1 else 'neutral'
        if sentiment_label == 'negative' and priority != 'high':
            priority = 'medium' if priority == 'low' else priority
    except Exception as e:
        logger.warning(f"Sentiment analysis failed: {e}")
        sentiment_label = 'neutral'

    # Detect category
    category = 'uncategorized'
    for cat in ['work', 'personal', 'school']:
        if cat in command:
            category = cat
            break

    # Extract description
    description = st.session_state.get('task_description', '')
    if not description:
        match = re.search(r'(?:description\s*[:|:]|.*?:)\s*(.+?)(?:\s*at\s*\d+|$)', command, re.IGNORECASE)
        if match:
            description = match.group(1).strip()[:50]

    with get_db_connection() as conn:
        cursor = conn.cursor()
        if 'add' in command:
            match = re.search(r'add\s+(.+?)(?:\s*description\s*[:|:]|:\s*(.+?))?(?:\s*at\s*(\d{1,2}(?::\d{2})?\s*[ap]m))?', command, re.IGNORECASE)
            if match:
                task_name = match.group(1).strip()
                desc = match.group(2).strip()[:50] if match.group(2) else description
                time = match.group(3) if match.group(3) else ''
                date = parse_date(command)
                cursor.execute("INSERT INTO tasks (user_id, name, date, time, description, status, priority, category) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", 
                              (user_id, task_name, date, parse_time(time), desc, 'pending', priority, category))
                conn.commit()
                st.session_state.task_description = ''
                return {'action': 'add', 'name': task_name, 'date': date, 'time': parse_time(time), 'description': desc, 'priority': priority, 'category': category}
        
        elif 'change' in command or 'update' in command:
            match = re.search(r'(change|update)\s+(.+?)\s+(?:to\s+(.+?))?(?:\s*description\s*[:|:]|:\s*(.+?))?(?:\s*at\s*(\d{1,2}(?::\d{2})?\s*[ap]m))?', command, re.IGNORECASE)
            if match:
                task_name = match.group(2).strip()
                new_time = match.group(5)
                new_desc = match.group(4).strip()[:50] if match.group(4) else None
                if new_time:
                    cursor.execute("UPDATE tasks SET time = ? WHERE name = ? AND user_id = ? AND status = 'pending'", 
                                   (parse_time(new_time), task_name, user_id))
                    conn.commit()
                    return {'action': 'update', 'name': task_name, 'time': parse_time(new_time)}
                elif new_desc:
                    cursor.execute("UPDATE tasks SET description = ? WHERE name = ? AND user_id = ? AND status = 'pending'", 
                                   (new_desc, task_name, user_id))
                    conn.commit()
                    return {'action': 'update_description', 'name': task_name, 'description': new_desc}
        
        elif 'mark' in command and 'done' in command:
            match = re.search(r'mark\s+(.+?)\s+as\s+done', command, re.IGNORECASE)
            if match:
                task_name = match.group(1).strip()
                cursor.execute("UPDATE tasks SET status = 'done' WHERE name = ? AND user_id = ? AND status = 'pending'", 
                               (task_name, user_id))
                conn.commit()
                return {'action': 'mark_done', 'name': task_name}
        
        elif 'list' in command or 'read' in command:
            category_filter = re.search(r'list\s+(work|personal|school)\s+tasks', command, re.IGNORECASE)
            query = "SELECT id, name, description, date, time, priority, category FROM tasks WHERE user_id = ? AND date >= ? AND status = 'pending'"
            params = [user_id, datetime.now().strftime('%Y-%m-%d')]
            if category_filter:
                query += " AND category = ?"
                params.append(category_filter.group(1))
            query += " ORDER BY CASE priority WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END, date, time"
            cursor.execute(query, params)
            tasks = [{'id': row['id'], 'name': row['name'], 'description': row['description'], 'date': row['date'], 'time': row['time'], 'priority': row['priority'], 'category': row['category']} for row in cursor.fetchall()]
            return {'action': 'list', 'tasks': tasks}
        
        elif 'use' in command and ('language' in command or 'hindi' in command or 'spanish' in command):
            lang_map = {'hindi': 'hi-IN', 'spanish': 'es-ES', 'english': 'en-US'}
            for lang, code in lang_map.items():
                if lang in command:
                    return {'action': 'set_language', 'lang': code}
        
        elif 'journal' in command:
            match = re.search(r'journal\s+(.+)', command)
            if match:
                content = match.group(1).strip()
                date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                cursor.execute("INSERT INTO journal (user_id, content, date) VALUES (?, ?, ?)", (user_id, content, date))
                conn.commit()
                return {'action': 'journal', 'content': content}
            if 'list journal' in command.lower():
                cursor.execute("SELECT content, date FROM journal WHERE user_id = ? ORDER BY date DESC", (user_id,))
                entries = [{'content': row['content'], 'date': row['date']} for row in cursor.fetchall()]
                return {'action': 'list_journal', 'entries': entries}
    
        return {'action': 'unknown'}

# Voice component HTML
voice_component_html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Voice Component</title>
    <style>
        body { font-family: Arial, sans-serif; padding: 10px; background-color: #f0f2f5; }
        button { padding: 10px 20px; margin: 5px; font-size: 16px; background-color: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background-color: #45a049; }
        select { padding: 10px; margin: 5px; font-size: 14px; border-radius: 4px; }
        #status { margin: 10px 0; font-weight: bold; color: #333; }
    </style>
    <script>
        let recognition;
        let isListening = false;
        let transcript = '';
        let context = 'CONTEXT_PLACEHOLDER';

        function startListening() {
            if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
                alert('Speech recognition not supported in this browser. Use a modern browser.');
                return;
            }

            recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
            recognition.lang = document.getElementById('lang-select').value || 'en-US';
            recognition.continuous = false;
            recognition.interimResults = false;

            recognition.onstart = function() {
                isListening = true;
                document.getElementById('status').innerText = 'Listening...';
                document.getElementById('start-btn').innerText = 'Stop Listening';
            };

            recognition.onresult = function(event) {
                transcript = event.results[0][0].transcript;
                document.getElementById('command').value = transcript;
                sendToStreamlit();
            };

            recognition.onerror = function(event) {
                document.getElementById('status').innerText = 'Error: ' + event.error;
                isListening = false;
                document.getElementById('start-btn').innerText = 'Start Listening';
            };

            recognition.onend = function() {
                isListening = false;
                document.getElementById('status').innerText = 'Ready';
                document.getElementById('start-btn').innerText = 'Start Listening';
            };

            recognition.start();
        }

        function toggleListening() {
            if (isListening) {
                recognition.stop();
            } else {
                startListening();
            }
        }

        function sendToStreamlit() {
            if (transcript) {
                window.parent.postMessage({
                    type: 'streamlit:setComponentValue',
                    value: { context: context, transcript: transcript }
                }, '*');
            }
        }
    </script>
</head>
<body>
    <select id="lang-select">
        <option value="en-US">English</option>
        <option value="hi-IN">Hindi</option>
        <option value="es-ES">Spanish</option>
    </select>
    <button id="start-btn" onclick="toggleListening()">Start Listening</button>
    <p id="status">Ready</p>
    <input type="hidden" id="command">
</body>
</html>
"""

# Streamlit Dashboard
st.set_page_config(page_title="OratoTask", layout="wide")
st.title("OratoTask Dashboard")

if not st.session_state.user_id:
    st.subheader("Login or Register")
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, password FROM users WHERE username = ?", (username,))
                result = cursor.fetchone()
                if result and bcrypt.checkpw(password.encode('utf-8'), result['password'].encode('utf-8')):
                    st.session_state.user_id = result['id']
                    st.session_state.username = username
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid credentials")
    
    with tab2:
        reg_username = st.text_input("Username", key="reg_username")
        reg_password = st.text_input("Password", type="password", key="reg_password")
        if st.button("Register"):
            hashed = bcrypt.hashpw(reg_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            with get_db_connection() as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (reg_username, hashed))
                    conn.commit()
                    st.success("User registered successfully! Please login.")
                except sqlite3.IntegrityError:
                    st.error("Username already exists")
else:
    st.subheader(f"Welcome, {st.session_state.username}!")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("Manage your tasks and journal entries below.")
    with col2:
        if st.button("Logout"):
            st.session_state.user_id = None
            st.session_state.username = None
            st.session_state.voice_command = ''
            st.session_state.voice_journal = ''
            st.session_state.voice_data = None
            st.session_state.task_description = ''
            st.success("Logged out successfully!")
            st.rerun()

    # Handle voice input
    voice_data = st.session_state.get('voice_data', None)
    if voice_data and isinstance(voice_data, dict) and 'context' in voice_data:
        if voice_data['context'] == 'command':
            st.session_state.voice_command = voice_data['transcript']
        elif voice_data['context'] == 'journal':
            st.session_state.voice_journal = voice_data['transcript']
        st.session_state.voice_data = None

    # Task Management Section
    with st.expander("Task Management", expanded=True):
        st.subheader("Add Task (Voice or Text)")
        col1, col2 = st.columns([2, 2])
        with col1:
            lang = st.selectbox("Select Language", ["English (en-US)", "Hindi (hi-IN)", "Spanish (es-ES)"], format_func=lambda x: x.split('(')[0].strip(), key="command_lang")
            lang_code = lang.split('(')[1].strip(')')
            voice_html = voice_component_html.replace('CONTEXT_PLACEHOLDER', 'command')
            voice_data = components.html(voice_html, height=150)
            if voice_data and isinstance(voice_data, dict) and 'context' in voice_data and voice_data['context'] == 'command':
                st.session_state.voice_data = voice_data
                st.rerun()
        with col2:
            command_text = st.text_input("Type command ( 'To add a meeting start with Add meeting to add a task start with ADD Task')", key="text_command")
            description_text = st.text_area("Task description (optional, max 50 chars)", key="description_text", max_chars=50, height=100)
            command = st.session_state.get('voice_command', '') or command_text
            if st.button("Process Command", key="process_command"):
                if command:
                    st.session_state.task_description = description_text
                    command_trans = translate_command(command, lang_code)
                    result = parse_command(command_trans, st.session_state.user_id)
                    sentiment_label = get_sentiment(command_trans)
                    if result['action'] == 'add':
                        message = f"Added {format_task_for_display(result)}"
                    elif result['action'] == 'update':
                        message = f"Task '{result['name']}' updated to {result['time']}."
                    elif result['action'] == 'update_description':
                        message = f"Task '{result['name']}' description updated to {result['description']}."
                    elif result['action'] == 'mark_done':
                        message = f"Task '{result['name']}' marked as done."
                    elif result['action'] == 'list':
                        if result['tasks']:
                            task_strings = [format_task_for_display(task) for task in result['tasks']]
                            message = f"Tasks: {', '.join(task_strings)}"
                        else:
                            message = "No tasks found."
                    elif result['action'] == 'set_language':
                        st.session_state.lang = result['lang']
                        message = f"Language set to {result['lang']}."
                    elif result['action'] == 'journal':
                        message = f"Journal entry saved: {result['content']}."
                    elif result['action'] == 'list_journal':
                        message = "Journal entries: " + "; ".join(
                            [f"{entry['date']}: {entry['content']}" for entry in result['entries']] if result['entries'] else "No entries found."
                        )
                    else:
                        message = "Unknown command. Try 'add task', 'list tasks', or 'journal'."
                    st.write(translate_response(message, lang_code))
                    if result.get('tasks'):
                        st.subheader("Listed Tasks")
                        for task in result['tasks']:
                            col_check, col_task = st.columns([1, 10])
                            with col_check:
                                if st.checkbox("", key=f"task_{task['id']}"):
                                    with get_db_connection() as conn:
                                        cursor = conn.cursor()
                                        cursor.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (task['id'],))
                                        conn.commit()
                                    st.success(f"Task '{task['name']}' completed!")
                                    st.rerun()
                            with col_task:
                                st.write(format_task_for_display(task))

        st.subheader("All Tasks")
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, description, date, time, priority, category, status FROM tasks WHERE user_id = ? ORDER BY date, time", (st.session_state.user_id,))
            all_tasks = cursor.fetchall()
        if all_tasks:
            task_strings = [format_task_for_display(task) for task in all_tasks if task['status'] == 'pending']
            if task_strings:
                st.write(", ".join(task_strings))
            for task in all_tasks:
                col1, col2, col3 = st.columns([1, 8, 1])
                with col1:
                    is_done = task['status'] == 'done'
                    if st.checkbox("", value=is_done, key=f"checkbox_{task['id']}"):
                        if not is_done:
                            with get_db_connection() as conn:
                                cursor = conn.cursor()
                                cursor.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (task['id'],))
                                conn.commit()
                            st.rerun()
                        else:
                            with get_db_connection() as conn:
                                cursor = conn.cursor()
                                cursor.execute("UPDATE tasks SET status = 'pending' WHERE id = ?", (task['id'],))
                                conn.commit()
                            st.rerun()
                with col2:
                    status_style = "text-decoration: line-through;" if task['status'] == 'done' else ""
                    priority_color = "red" if task['priority'] == 'high' else "orange" if task['priority'] == 'medium' else "green"
                    st.markdown(f"<span style='{status_style} color:{priority_color}'>{format_task_for_display(task)}</span>", unsafe_allow_html=True)
                with col3:
                    if st.button("🗑", key=f"delete_{task['id']}"):
                        with get_db_connection() as conn:
                            cursor = conn.cursor()
                            cursor.execute("DELETE FROM tasks WHERE id = ?", (task['id'],))
                            conn.commit()
                        st.success("Task deleted!")
                        st.rerun()
        else:
            st.info("No tasks found.")

    # Journaling Section
    with st.expander("Journaling", expanded=True):
        st.subheader("Add Journal Entry (Voice or Text)")
        col1, col2 = st.columns([2, 2])
        with col1:
            journal_lang = st.selectbox("Journal Language", ["English (en-US)", "Hindi (hi-IN)", "Spanish (es-ES)"], format_func=lambda x: x.split('(')[0].strip(), key="journal_lang")
            journal_lang_code = journal_lang.split('(')[1].strip(')')
            journal_voice_html = voice_component_html.replace('CONTEXT_PLACEHOLDER', 'journal')
            journal_voice_data = components.html(journal_voice_html, height=150)
            if journal_voice_data and isinstance(journal_voice_data, dict) and 'context' in journal_voice_data and journal_voice_data['context'] == 'journal':
                st.session_state.voice_data = journal_voice_data
                st.rerun()
        with col2:
            journal_text = st.text_area("Write your thoughts", key="text_journal", height=100)
            journal_content = st.session_state.get('voice_journal', '') or journal_text
            if st.button("Save Journal", key="save_journal"):
                if journal_content:
                    content_trans = translate_command(journal_content, journal_lang_code)
                    date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    with get_db_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute("INSERT INTO journal (user_id, content, date) VALUES (?, ?, ?)", (st.session_state.user_id, content_trans, date))
                        conn.commit()
                    st.success(translate_response(f"Journal entry saved: {content_trans}.", journal_lang_code))
                    st.session_state.voice_journal = ''
                    st.rerun()

        st.subheader("Journal History")
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT content, date FROM journal WHERE user_id = ? ORDER BY date DESC", (st.session_state.user_id,))
            entries = [{'content': row['content'], 'date': row['date']} for row in cursor.fetchall()]
        if entries:
            for entry in entries:
                sentiment = get_sentiment(entry['content'])
                st.markdown(f"- {entry['date']}: {entry['content']} (Sentiment: {sentiment})")
        else:
            st.info("No journal entries found.")

    # Calendar Section
    with st.expander("Calendar View", expanded=True):
        st.subheader("Task Calendar")
        col1, col2, col3 = st.columns([2, 2, 2])
        with col1:
            year = st.number_input("Year", min_value=2020, max_value=2030, value=datetime.now().year, key="calendar_year")
        with col2:
            month = st.selectbox("Month", list(range(1, 13)), index=datetime.now().month-1, key="calendar_month")
        with col3:
            sort_by = st.radio("Sort by", ["Priority", "Time"], key="calendar_sort")
        if st.button("Show Calendar", key="show_calendar"):
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, name, description, date, time, priority, category, status FROM tasks WHERE user_id = ?", (st.session_state.user_id,))
                all_tasks = [{'id': row['id'], 'name': row['name'], 'description': row['description'], 'date': row['date'], 'time': row['time'], 'priority': row['priority'], 'category': row['category'], 'status': row['status']} for row in cursor.fetchall()]
            task_dict = {}
            for task in all_tasks:
                date = task['date']
                if date not in task_dict:
                    task_dict[date] = []
                task_dict[date].append(task)
            for date in task_dict:
                if sort_by == "Priority":
                    task_dict[date].sort(key=lambda x: (['high', 'medium', 'low'].index(x['priority']), parse_time(x['time']) if x['time'] != 'anytime' else '23:59'))
                else:
                    task_dict[date].sort(key=lambda x: (parse_time(x['time']) if x['time'] != 'anytime' else '23:59', ['high', 'medium', 'low'].index(x['priority'])))
            cal = calendar.monthcalendar(year, month)
            st.write(f"Calendar for {calendar.month_name[month]} {year}")
            cal_html = "<table style='width:100%; border-collapse:collapse; background-color:#000000; color:#ffffff;'>"
            cal_html += "<tr style='background-color:#333333;'><th>Sun</th><th>Mon</th><th>Tue</th><th>Wed</th><th>Thu</th><th>Fri</th><th>Sat</th></tr>"
            for week in cal:
                cal_html += "<tr>"
                for day in week:
                    if day == 0:
                        cal_html += "<td style='border:1px solid #444444; padding:8px; height:100px; vertical-align:top; background-color:#000000;'></td>"
                    else:
                        date_str = f"{year}-{month:02d}-{day:02d}"
                        tasks_for_day = task_dict.get(date_str, [])
                        task_html = "".join(
                            f"<div style='color:{'red' if task['priority'] == 'high' else 'orange' if task['priority'] == 'medium' else 'green'}; text-decoration:{'line-through' if task['status'] == 'done' else ''};font-size:12px;'>{format_task_for_display(task)} {'✓' if task['status'] == 'done' else ''}</div>"
                            for task in tasks_for_day
                        )
                        cal_html += f"<td style='border:1px solid #444444; padding:8px; height:100px; vertical-align:top; background-color:#000000;'>{day}<div>{task_html}</div></td>"
                cal_html += "</tr>"
            cal_html += "</table>"
            st.markdown(cal_html, unsafe_allow_html=True)