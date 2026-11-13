
import os
import re
import secrets
import mysql.connector

from flask import (
    Flask, render_template, request,
    redirect, url_for, session, flash
)

from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from google import genai

# --------------------------------------------------
# 1. LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------

load_dotenv()

app = Flask(__name__)

app.secret_key = os.getenv("SECRET_KEY", "presentiq-secret-key")

UPLOAD_FOLDER = "recordings"
ALLOWED_EXTENSIONS = {"webm", "mp4", "mov", "wav", "mp3"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --------------------------------------------------
# 2. GEMINI CONFIGURATION
# --------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

client = None

if GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        print("Gemini API configured successfully.")
    except Exception as e:
        print("Gemini initialization failed:", e)
else:
    print("Gemini API key not found. Fallback mode will be used.")


# --------------------------------------------------
# 3. MYSQL DATABASE CONNECTION
# --------------------------------------------------

def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "localhost"),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", "presentiq")
    )


# --------------------------------------------------
# 4. HELPER FUNCTIONS
# --------------------------------------------------

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def fallback_feedback(speech_text):
    """
    Rule-based feedback when Gemini is unavailable.
    This evaluates speech text, not video or voice delivery.
    """

    text = (speech_text or "").strip()

    if not text:
        return 0, (
            "Fallback Feedback\n\n"
            "No speech text was provided for evaluation.\n\n"
            "Please enter your presentation speech and try again."
        )

    words = re.findall(r"\b[\w']+\b", text)
    word_count = len(words)

    sentences = [
        s.strip()
        for s in re.split(r"[.!?]+", text)
        if s.strip()
    ]

    sentence_count = max(len(sentences), 1)
    avg_sentence_length = word_count / sentence_count

    lower_text = text.lower()

    transition_words = [
        "firstly", "secondly", "finally", "next",
        "therefore", "however", "moreover",
        "furthermore", "in conclusion", "for example",
        "in addition", "to summarize"
    ]

    transitions_found = [
        word for word in transition_words
        if word in lower_text
    ]

    filler_words = [
        "um", "uh", "actually", "basically",
        "like", "you know"
    ]

    filler_count = sum(
        len(re.findall(r"\b" + re.escape(word) + r"\b", lower_text))
        for word in filler_words
    )

    score = 50
    strengths = []
    improvements = []

    # Speech length evaluation
    if word_count >= 150:
        score += 15
        strengths.append("Good speech length for a detailed presentation.")
    elif word_count >= 75:
        score += 10
        strengths.append("Your speech contains a reasonable amount of content.")
    else:
        improvements.append(
            "Expand your presentation with more explanations and examples."
        )

    # Sentence structure evaluation
    if sentence_count >= 5:
        score += 10
        strengths.append("Your speech contains multiple sentences.")
    else:
        improvements.append(
            "Use more complete sentences to organize your ideas."
        )

    # Sentence length evaluation
    if 8 <= avg_sentence_length <= 25:
        score += 10
        strengths.append("Your sentence length is reasonably balanced.")
    elif avg_sentence_length > 25:
        improvements.append(
            "Break long sentences into shorter, clearer sentences."
        )
    else:
        improvements.append(
            "Develop your ideas with more complete explanations."
        )

    # Transition word evaluation
    if transitions_found:
        score += 10
        strengths.append(
            "You use connecting words to organize your presentation."
        )
    else:
        improvements.append(
            "Use transition words such as Firstly, However and Finally."
        )

    # Filler word evaluation
    if filler_count == 0:
        score += 5
        strengths.append(
            "No common filler words were detected in your speech text."
        )
    else:
        improvements.append(
            f"Review the use of filler words. "
            f"{filler_count} possible filler-word occurrences were detected."
        )

    score = max(0, min(score, 100))

    if not strengths:
        strengths.append("You have started building your presentation speech.")

    if not improvements:
        improvements.append(
            "Continue practising to improve your presentation skills."
        )

    feedback = f"""
FALLBACK PRESENTATION FEEDBACK

Note: Gemini AI was temporarily unavailable.
This feedback was generated using PresentIQ's rule-based evaluation system.

OVERALL SCORE: {score}/100

SPEECH ANALYSIS
- Total words: {word_count}
- Number of sentences: {sentence_count}
- Average words per sentence: {avg_sentence_length:.1f}
- Possible filler-word occurrences: {filler_count}
- Transition words detected: {len(transitions_found)}

STRENGTHS
{chr(10).join('- ' + item for item in strengths)}

AREAS FOR IMPROVEMENT
{chr(10).join('- ' + item for item in improvements)}

RECOMMENDATIONS
1. Practise your presentation regularly.
2. Speak clearly and use a natural pace.
3. Organize your speech with an introduction, main points and conclusion.
4. Use examples to explain important concepts.
5. Maintain confidence while presenting.

Keep practising! Every presentation is an opportunity to improve.
"""

    return score, feedback.strip()


# --------------------------------------------------
# 5. GEMINI AI FEEDBACK
# --------------------------------------------------

def generate_gemini_feedback(speech_text):

    if not client:
        raise Exception("Gemini API is not configured.")

    prompt = f"""
You are an expert presentation coach.

Evaluate the following presentation speech.

Give feedback on:
1. Content quality
2. Clarity
3. Organization
4. Vocabulary
5. Grammar
6. Presentation improvements

Give practical and beginner-friendly suggestions.

Start your response with a score in this exact format:

SCORE: 85

Then provide:

STRENGTHS:
AREAS FOR IMPROVEMENT:
RECOMMENDATIONS:

Evaluate only the supplied text.
Do not claim to have evaluated voice, facial expressions or body language.

PRESENTATION SPEECH:

{speech_text}
"""

    last_error = None

    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )

            feedback = response.text

            if not feedback:
                raise Exception("Gemini returned an empty response.")

            match = re.search(
                r"SCORE\s*:\s*(\d{1,3})",
                feedback,
                re.IGNORECASE
            )

            score = 70

            if match:
                score = max(0, min(int(match.group(1)), 100))

            return score, feedback.strip()

        except Exception as e:
            last_error = e
            print(f"Gemini attempt {attempt + 1} failed: {e}")

    raise Exception(str(last_error))


# --------------------------------------------------
# 6. HOME PAGE
# --------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# --------------------------------------------------
# 7. REGISTER
# --------------------------------------------------

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        if not name or not email or not password:
            flash("Please fill in all fields.")
            return redirect(url_for("register"))

        hashed_password = generate_password_hash(password)

        try:
            db = get_db_connection()
            cursor = db.cursor()

            cursor.execute(
                """
                INSERT INTO users (name, email, password)
                VALUES (%s, %s, %s)
                """,
                (name, email, hashed_password)
            )

            db.commit()

            cursor.close()
            db.close()

            flash("Registration successful. Please login.")
            return redirect(url_for("login"))

        except mysql.connector.IntegrityError:
            flash("Email already exists. Please login.")

        except Exception as e:
            print("Registration error:", e)
            flash("Registration failed. Please try again.")

        return redirect(url_for("register"))

    return render_template("register.html")


# --------------------------------------------------
# 8. LOGIN
# --------------------------------------------------

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        try:
            db = get_db_connection()
            cursor = db.cursor(dictionary=True)

            cursor.execute(
                "SELECT * FROM users WHERE email = %s",
                (email,)
            )

            user = cursor.fetchone()

            cursor.close()
            db.close()

            if user and check_password_hash(user["password"], password):

                session["user_id"] = user["id"]
                session["user_name"] = user["name"]

                return redirect(url_for("dashboard"))

            flash("Invalid email or password.")
            return redirect(url_for("login"))

        except Exception as e:
            print("Login error:", e)
            flash("Unable to login. Check database connection.")
            return redirect(url_for("login"))

    return render_template("login.html")


# --------------------------------------------------
# 9. DASHBOARD
# --------------------------------------------------

@app.route("/dashboard")
def dashboard():

    if "user_id" not in session:
        return redirect(url_for("login"))

    try:
        db = get_db_connection()
        cursor = db.cursor(dictionary=True)

        cursor.execute(
            """
            SELECT *
            FROM presentations
            WHERE user_id = %s
            ORDER BY created_at DESC
            """,
            (session["user_id"],)
        )

        presentations = cursor.fetchall()

        cursor.close()
        db.close()

        total_presentations = len(presentations)

        if total_presentations:
            average_score = round(
                sum(int(p["score"] or 0) for p in presentations)
                / total_presentations
            )
        else:
            average_score = 0

        return render_template(
            "dashboard.html",
            presentations=presentations,
            total_presentations=total_presentations,
            average_score=average_score,
            user_name=session.get("user_name")
        )

    except Exception as e:
        print("Dashboard error:", e)
        flash("Unable to load dashboard.")
        return render_template(
            "dashboard.html",
            presentations=[],
            total_presentations=0,
            average_score=0,
            user_name=session.get("user_name")
        )


# --------------------------------------------------
# 10. PRACTICE PAGE
# --------------------------------------------------

@app.route("/practice", methods=["GET", "POST"])
def practice():

    if "user_id" not in session:
        return redirect(url_for("login"))

    if request.method == "GET":
        return render_template("practice.html")

    title = request.form.get("title", "").strip()
    speech_text = request.form.get("speech_text", "").strip()

    if not title:
        title = "My Presentation"

    recording = request.files.get("recording")
    recording_filename = None

    # Save uploaded recording
    if recording and recording.filename:

        if not allowed_file(recording.filename):
            flash("Unsupported recording file format.")
            return redirect(url_for("practice"))

        original_filename = secure_filename(recording.filename)

        extension = original_filename.rsplit(".", 1)[1].lower()

        recording_filename = (
            secrets.token_hex(12) + "." + extension
        )

        recording.save(
            os.path.join(
                app.config["UPLOAD_FOLDER"],
                recording_filename
            )
        )

    # Generate feedback
    try:
        if not speech_text:
            raise Exception("Speech text was not provided.")

        score, feedback = generate_gemini_feedback(speech_text)

        feedback_source = "Gemini AI"
        print("Gemini feedback generated successfully.")

    except Exception as e:

        print("Switching to fallback feedback:", e)

        score, feedback = fallback_feedback(speech_text)

        feedback_source = "Fallback System"

        if speech_text:
            print("Fallback feedback generated successfully.")

    # Save presentation in MySQL
    try:
        db = get_db_connection()
        cursor = db.cursor()

        cursor.execute(
            """
            INSERT INTO presentations
            (user_id, title, speech_text, recording, score, feedback)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                session["user_id"],
                title,
                speech_text,
                recording_filename,
                score,
                feedback
            )
        )

        presentation_id = cursor.lastrowid

        db.commit()

        cursor.close()
        db.close()

        flash(f"Feedback generated using {feedback_source}.")

        return redirect(
            url_for("result", presentation_id=presentation_id)
        )

    except Exception as e:

        print("Database save error:", e)
        flash("Unable to save presentation result.")

        return redirect(url_for("practice"))


# --------------------------------------------------
# 11. RESULT PAGE
# --------------------------------------------------

@app.route("/result/<int:presentation_id>")
def result(presentation_id):

    if "user_id" not in session:
        return redirect(url_for("login"))

    try:
        db = get_db_connection()
        cursor = db.cursor(dictionary=True)

        cursor.execute(
            """
            SELECT *
            FROM presentations
            WHERE id = %s AND user_id = %s
            """,
            (presentation_id, session["user_id"])
        )

        presentation = cursor.fetchone()

        cursor.close()
        db.close()

        if not presentation:
            flash("Presentation result not found.")
            return redirect(url_for("dashboard"))

        return render_template(
            "result.html",
            presentation=presentation
        )

    except Exception as e:
        print("Result page error:", e)
        flash("Unable to load presentation result.")
        return redirect(url_for("dashboard"))


# --------------------------------------------------
# 12. LOGOUT
# --------------------------------------------------

@app.route("/logout")
def logout():

    session.clear()

    flash("You have been logged out.")

    return redirect(url_for("login"))


# --------------------------------------------------
# 13. FILE SIZE ERROR
# --------------------------------------------------

@app.errorhandler(413)
def file_too_large(error):

    flash("Recording is too large. Maximum size is 100 MB.")

    return redirect(url_for("practice"))


# --------------------------------------------------
# 14. RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
