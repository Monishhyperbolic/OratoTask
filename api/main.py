from fastapi import FastAPI, HTTPException, Response, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sqlite3
from datetime import datetime, timedelta
import re
from transformers import pipeline
from googletrans import Translator
import joblib
import nltk
from nltk.corpus import stopwords
import calendar
from dateutil.parser import parse
import bcrypt
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
translator = Translator()
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Load priority model
model_path = '/workspaces/OratoTask/api/priority_model.joblib'
if not os.path.exists(model_path):
    logger.error(f"Priority model not found at {model_path}")
    raise FileNotFoundError(f"Priority model not found at {model_path}")
priority_model = joblib.load(model_path)

# Database connection function
def get_db_connection():
    db_path = '/workspaces/OratoTask/tasks.db'
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

class User(BaseModel):
    username: str
    password: str

class Command(BaseModel):
    command: str
    lang: str = 'en-US'

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

def get_current_user(request: Request):
    user_id = request.cookies.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        return int(user_id)
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid user ID")

def parse_command(command: str, user_id: int):
    command = command.lower()
    clean_command = preprocess_text(command)

    # Handle voice-based login
    login_match = re.search(r'login as user (\w+) with password (\w+)', command, re.IGNORECASE)
    if login_match:
        username = login_match.group(1)
        password = login_match.group(2)
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, password FROM users WHERE username = ?", (username,))
            result = cursor.fetchone()
            if result and bcrypt.checkpw(password.encode(), result['password'].encode()):
                return {'action': 'login', 'user_id': result['id'], 'username': username}
        return {'action': 'login_failed'}

    # ML-based priority detection
    try:
        priority = priority_model.predict([clean_command])[0]
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
        sentiment = sentiment_analyzer(command)[0]
        sentiment_label = 'negative' if sentiment['label'] == 'NEGATIVE' else 'positive' if sentiment['score'] > 0.7 else 'neutral'
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

@app.get("/")
async def root():
    return {"message": "Welcome to AccessiVoice Planner API. Access the frontend at /public/index.html or use API endpoints like /login, /register, /process-command."}

@app.post("/register")
async def register(user: User):
    hashed = bcrypt.hashpw(user.password.encode(), bcrypt.gensalt()).decode()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (user.username, hashed))
            conn.commit()
            return {"message": "User registered successfully"}
        except sqlite3.IntegrityError:
            raise HTTPException(status_code=400, detail="Username already exists")

@app.post("/login")
async def login(user: User, response: Response):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, password FROM users WHERE username = ?", (user.username,))
        result = cursor.fetchone()
        if result and bcrypt.checkpw(user.password.encode(), result['password'].encode()):
            response.set_cookie(key="user_id", value=str(result['id']), httponly=True, samesite='lax')
            return {"message": "Login successful", "username": user.username}
        raise HTTPException(status_code=401, detail="Invalid credentials")

@app.post("/logout")
async def logout(response: Response):
    response.delete_cookie("user_id")
    return {"message": "Logged out successfully"}

@app.post("/process-command")
async def process_command(cmd: Command, response: Response, user_id: int = Depends(get_current_user)):
    command = translate_command(cmd.command, cmd.lang)
    result = parse_command(command, user_id)
    try:
        sentiment = sentiment_analyzer(command)[0]
        sentiment_label = 'negative' if sentiment['label'] == 'NEGATIVE' else 'positive' if sentiment['score'] > 0.7 else 'neutral'
    except Exception as e:
        logger.warning(f"Sentiment analysis failed: {e}")
        sentiment_label = 'neutral'

    if result['action'] == 'login':
        response.set_cookie(key="user_id", value=str(result['user_id']), httponly=True, samesite='lax')
        message = f"Logged in as {result['username']}."
    elif result['action'] == 'login_failed':
        message = "Invalid login credentials."
    elif result['action'] == 'add':
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
            message = "Here are your tasks, prioritized: " + "; ".join(
                [f"{task['name']} on {task['date']} at {task['time'] or 'anytime'} ({task['priority']} priority, {task['category']})" 
                 for task in result['tasks']]
            )
            high_priority = next((task for task in result['tasks'] if task['priority'] == 'high'), None)
            if high_priority:
                message += f" Focus on this high-priority task: {high_priority['name']}."
        else:
            message = "No tasks found."
    elif result['action'] == 'set_language':
        message = f"Language set to {result['lang']}."
    else:
        message = "Sorry, I didn't understand. Try 'add task', 'list tasks', or 'journal'."

    return {"message": translate_response(message, cmd.lang), "tasks": result.get('tasks', []), "lang": result.get('lang', cmd.lang)}

@app.post("/journal-entry")
async def journal_entry(cmd: Command, user_id: int = Depends(get_current_user)):
    content = translate_command(cmd.command, cmd.lang)
    date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO journal (user_id, content, date) VALUES (?, ?, ?)", (user_id, content, date))
        conn.commit()
        cursor.execute("SELECT content, date FROM journal WHERE user_id = ? ORDER BY date DESC LIMIT 5", (user_id,))
        entries = [{'content': row['content'], 'date': row['date']} for row in cursor.fetchall()]
    message = f"Journal entry saved: {content}. Say 'list journal' to hear recent entries."
    if 'list journal' in content.lower():
        message = "Recent journal entries: " + "; ".join([f"{entry['date']}: {entry['content']}" for entry in entries])
    return {"message": translate_response(message, cmd.lang), "journalEntries": entries}

@app.get("/calendar/{year}/{month}")
async def get_calendar(year: int, month: int, user_id: int = Depends(get_current_user)):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name, date, time, priority, category FROM tasks WHERE user_id = ? AND status = 'pending'", (user_id,))
        tasks = [{'name': row['name'], 'date': row['date'], 'time': row['time'], 'priority': row['priority'], 'category': row['category']} for row in cursor.fetchall()]
    
    task_dict = {}
    for task in tasks:
        date = task['date']
        if date not in task_dict:
            task_dict[date] = []
        task_dict[date].append(task)

    cal = calendar.monthcalendar(year, month)
    html = f"""
    <table class='w-full border-collapse' role='grid' aria-label='Calendar for {calendar.month_name[month]} {year}'>
        <thead>
            <tr class='bg-gray-200'>
                <th class='border p-2' scope='col'>Sun</th>
                <th class='border p-2' scope='col'>Mon</th>
                <th class='border p-2' scope='col'>Tue</th>
                <th class='border p-2' scope='col'>Wed</th>
                <th class='border p-2' scope='col'>Thu</th>
                <th class='border p-2' scope='col'>Fri</th>
                <th class='border p-2' scope='col'>Sat</th>
            </tr>
        </thead>
        <tbody>
    """
    for week in cal:
        html += "<tr>"
        for day in week:
            if day == 0:
                html += "<td class='border p-2'></td>"
            else:
                date_str = f"{year}-{month:02d}-{day:02d}"
                tasks_for_day = task_dict.get(date_str, [])
                task_html = "".join(
                    f"<div class='{task['priority'] == 'high' and 'text-red-600' or task['priority'] == 'medium' and 'text-yellow-600' or 'text-green-600'}'>{task['name']} ({task['time'] or 'anytime'})</div>"
                    for task in tasks_for_day
                )
                html += f"<td class='border p-2' role='gridcell' aria-label='Day {day}'>{day}<div>{task_html}</div></td>"
        html += "</tr>"
    html += "</tbody></table>"
    return {"calendar_html": html}