import streamlit as st
import sqlite3
import re
from datetime import datetime, timedelta
from textblob import TextBlob
from googletrans import Translator
import joblib
import nltk
from nltk.corpus import stopwords
import calendar
from dateutil.parser import parse
import bcrypt
import logging
import os
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Initialize translator
translator = Translator()

# Load priority model and vectorizer
model_path = 'api/priority_model.joblib'
vectorizer_path = 'api/vectorizer.joblib'
if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    logger.error(f"Model or vectorizer not found at {model_path} or {vectorizer_path}")
    raise FileNotFoundError(f"Model or vectorizer not found")
priority_model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Database connection
def get_db_connection():
    db_path = 'tasks.db'
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

# Initialize database
with get_db_connection() as conn:
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users
                      (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS tasks
                      (id INTEGER PRIMARY KEY, user_id INTEGER, name TEXT, date TEXT, time TEXT, status TEXT, priority TEXT, category TEXT,
                       FOREIGN KEY(user_id) REFERENCES users(id))''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS journal
                      (id INTEGER PRIMARY KEY, user_id INTEGER, content TEXT, date TEXT,
                       FOREIGN KEY(user_id) REFERENCES users(id))''')
    conn.commit()

# Streamlit session state
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'username' not in st.session_state:
    st.session_state.username = None
if 'lang' not in st.session_state:
    st.session_state.lang = 'en-US'

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

def translate_command(command: str, lang: str):
    if lang != 'en-US':
        try:
            return translator.translate(command, src=lang.split('-')[0], dest='en').text
        except Exception as e:
            logger.warning(f"Translation failed: {e}")
            return command
    return command

def translate_response(message: str, lang: str):
    if lang != 'en-US':
        try:
            return translator.translate(message, src='en', dest=lang.split('-')[0]).text
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

def parse_command(command: str, user_id: int):
    command = command.lower()
    clean_command = preprocess_text(command)

    # ML-based priority detection
    try:
        X_vec = vectorizer.transform([clean_command])
        priority = priority_model.predict(X_vec)[0]
    except Exception as e:
        logger.error(f"Priority model prediction failed: {e}")
        priority = 'low'

    # Keyword-based fallback
    priority_keywords = {'urgent': 'high', 'important': 'high', 'asap': 'high', 'critical': 'high', 'soon': 'medium', 'today': 'medium'}
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

    with get_db_connection() as conn:
        cursor = conn.cursor()
        if 'add' in command:
            match = re.search(r'add (.+?)(?: at (\d{1,2}(?::\d{2})? ?[ap]m))?', command)
            if match:
                task_name = match.group(1)
                time = match.group(2) if match.group(2) else ''
                date = parse_date(command)
                cursor.execute("INSERT INTO tasks (user_id, name, date, time, status, priority, category) VALUES (?, ?, ?, ?, ?, ?, ?)", 
                              (user_id, task_name, date, time, 'pending', priority, category))
                conn.commit()
                return {'action': 'add', 'task_name': task_name, 'date': date, 'time': time, 'priority': priority, 'category': category}
        
        elif 'change' in command or 'update' in command:
            match = re.search(r'(change|update) (.+?) to (\d{1,2}(?::\d{2})? ?[ap]m)', command)
            if match:
                task_name = match.group(2)
                new_time = match.group(3)
                cursor.execute("UPDATE tasks SET time = ? WHERE name = ? AND user_id = ? AND status = 'pending'", (new_time, task_name, user_id))
                conn.commit()
                return {'action': 'update', 'task_name': task_name, 'time': new_time}
        
        elif 'mark' in command and 'done' in command:
            match = re.search(r'mark (.+?) as done', command)
            if match:
                task_name = match.group(1)
                cursor.execute("UPDATE tasks SET status = 'done' WHERE name = ? AND user_id = ? AND status = 'pending'", (task_name, user_id))
                conn.commit()
                return {'action': 'mark_done', 'task_name': task_name}
        
        elif 'list' in command or 'read' in command:
            category_filter = re.search(r'list (work|personal|school) tasks', command)
            query = "SELECT name, date, time, priority, category FROM tasks WHERE user_id = ? AND date >= ? AND status = 'pending'"
            params = [user_id, datetime.now().strftime('%Y-%m-%d')]
            if category_filter:
                query += " AND category = ?"
                params.append(category_filter.group(1))
            query += " ORDER BY CASE priority WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END, date, time"
            cursor.execute(query, params)
            tasks = [{'name': row['name'], 'date': row['date'], 'time': row['time'], 'priority': row['priority'], 'category': row['category']} for row in cursor.fetchall()]
            return {'action': 'list', 'tasks': tasks}
        
        elif 'use' in command and ('language' in command or 'hindi' in command or 'spanish' in command):
            lang_map = {'hindi': 'hi-IN', 'spanish': 'es-ES', 'english': 'en-US'}
            for lang, code in lang_map.items():
                if lang in command:
                    return {'action': 'set_language', 'lang': code}
    
    return {'action': 'unknown'}

# Streamlit UI
st.title("AccessiVoice Planner")

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
                if result and bcrypt.checkpw(password.encode(), result['password'].encode()):
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
            hashed = bcrypt.hashpw(reg_password.encode(), bcrypt.gensalt()).decode()
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
    if st.button("Logout"):
        st.session_state.user_id = None
        st.session_state.username = None
        st.success("Logged out successfully!")
        st.rerun()
    
    st.subheader("Command Input")
    command = st.text_input("Enter command (e.g., 'Add meeting on July 20, 2025 at 3 PM', 'List tasks', 'Journal I’m excited')")
    lang = st.selectbox("Language", ["English (en-US)", "Hindi (hi-IN)", "Spanish (es-ES)"], format_func=lambda x: x.split('(')[0].strip())
    lang_code = lang.split('(')[1].strip(')')
    
    if st.button("Process Command"):
        if command:
            command_trans = translate_command(command, lang_code)
            result = parse_command(command_trans, st.session_state.user_id)
            try:
                blob = TextBlob(command_trans)
                sentiment_score = blob.sentiment.polarity
                sentiment_label = 'negative' if sentiment_score < -0.1 else 'positive' if sentiment_score > 0.1 else 'neutral'
            except Exception as e:
                logger.warning(f"Sentiment analysis failed: {e}")
                sentiment_label = 'neutral'

            if result['action'] == 'add':
                message = f"Task {result['task_name']} added for {result['date']} at {result['time'] or 'anytime'} with {result['priority']} priority in {result['category']}."
                if sentiment_label == 'negative':
                    message += " Sounds like a busy day! Take a deep breath, we've got this."
                elif sentiment_label == 'positive':
                    message += " Awesome, you're killing it!"
            elif result['action'] == 'update':
                message = f"Task {result['task_name']} updated to {result['time']}."
            elif result['action'] == 'mark_done':
                message = f"Task {result['task_name']} marked as done."
            elif result['action'] == 'list':
                if result['tasks']:
                    message = "Here are your tasks: " + "; ".join(
                        [f"{task['name']} on {task['date']} at {task['time'] or 'anytime'} ({task['priority']} priority, {task['category']})" 
                         for task in result['tasks']]
                    )
                    high_priority = next((task for task in result['tasks'] if task['priority'] == 'high'), None)
                    if high_priority:
                        message += f" Focus on this high-priority task: {high_priority['name']}."
                else:
                    message = "No tasks found."
            elif result['action'] == 'set_language':
                st.session_state.lang = result['lang']
                message = f"Language set to {result['lang']}."
            else:
                message = "Sorry, I didn’t understand. Try 'add task', 'list tasks', or 'journal'."

            st.write(translate_response(message, lang_code))
            if result.get('tasks'):
                st.subheader("Tasks")
                for task in result['tasks']:
                    st.write(f"- {task['name']} on {task['date']} at {task['time'] or 'anytime'} ({task['priority']} priority, {task['category']})")

    st.subheader("Journal Entry")
    journal_content = st.text_area("Write your thoughts")
    if st.button("Save Journal"):
        if journal_content:
            content_trans = translate_command(journal_content, lang_code)
            date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO journal (user_id, content, date) VALUES (?, ?, ?)", (st.session_state.user_id, content_trans, date))
                conn.commit()
                cursor.execute("SELECT content, date FROM journal WHERE user_id = ? ORDER BY date DESC LIMIT 5", (st.session_state.user_id,))
                entries = [{'content': row['content'], 'date': row['date']} for row in cursor.fetchall()]
            message = f"Journal entry saved: {content_trans}. View recent entries below."
            if 'list journal' in content_trans.lower():
                message = "Recent journal entries: " + "; ".join([f"{entry['date']}: {entry['content']}" for entry in entries])
            st.write(translate_response(message, lang_code))
            st.subheader("Recent Journal Entries")
            for entry in entries:
                st.write(f"{entry['date']}: {entry['content']}")

    st.subheader("Calendar")
    year = st.number_input("Year", min_value=2020, max_value=2030, value=datetime.now().year)
    month = st.selectbox("Month", list(range(1, 13)), index=datetime.now().month-1)
    if st.button("Show Calendar"):
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name, date, time, priority, category FROM tasks WHERE user_id = ? AND status = 'pending'", (st.session_state.user_id,))
            tasks = [{'name': row['name'], 'date': row['date'], 'time': row['time'], 'priority': row['priority'], 'category': row['category']} for row in cursor.fetchall()]
        
        task_dict = {}
        for task in tasks:
            date = task['date']
            if date not in task_dict:
                task_dict[date] = []
            task_dict[date].append(task)

        cal = calendar.monthcalendar(year, month)
        st.write(f"Calendar for {calendar.month_name[month]} {year}")
        cal_html = "<table style='width:100%; border-collapse:collapse;'>"
        cal_html += "<tr><th>Sun</th><th>Mon</th><th>Tue</th><th>Wed</th><th>Thu</th><th>Fri</th><th>Sat</th></tr>"
        for week in cal:
            cal_html += "<tr>"
            for day in week:
                if day == 0:
                    cal_html += "<td style='border:1px solid #ddd; padding:8px;'></td>"
                else:
                    date_str = f"{year}-{month:02d}-{day:02d}"
                    tasks_for_day = task_dict.get(date_str, [])
                    task_html = "".join(
                        f"<div style='color:{'red' if task['priority'] == 'high' else 'orange' if task['priority'] == 'medium' else 'green'}'>{task['name']} ({task['time'] or 'anytime'})</div>"
                        for task in tasks_for_day
                    )
                    cal_html += f"<td style='border:1px solid #ddd; padding:8px;'>{day}<div>{task_html}</div></td>"
            cal_html += "</tr>"
        cal_html += "</table>"
        st.markdown(cal_html, unsafe_allow_html=True)