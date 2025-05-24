from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import docx
import fitz  # PyMuPDF
import time
from groq import Groq
import httpx
from neo4j import GraphDatabase
from dotenv import load_dotenv
import uuid
import logging
from logging.handlers import RotatingFileHandler
import json
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from difflib import SequenceMatcher  # For string similarity

# === Config ===
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY', 'your api key')
NEO4J_URI = os.getenv('NEO4J_URI', 'ur uri')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME', 'your user')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'your pass')

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not set in environment variables.")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

# Setup logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'legal_chatbot.log')
handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5)  # 5MB per file, keep 5 backups
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.basicConfig(
    level=logging.DEBUG,
    handlers=[
        handler,
        logging.StreamHandler()  # Continue logging to console
    ]
)
logger = logging.getLogger(__name__)

# Initialize Neo4j driver
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(Exception),
)
def init_neo4j_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Initialize clients
try:
    groq_client = Groq(api_key=GROQ_API_KEY)
    neo4j_driver = init_neo4j_driver()
except Exception as e:
    logger.error(f"Failed to initialize clients: {str(e)}")
    raise ValueError(f"Cannot connect to Neo4j at {NEO4J_URI}.")

UPLOAD_FOLDER = 'Uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB limit

# Load ground truth data
try:
    with open('ground_truth.json', 'r') as f:
        GROUND_TRUTH = json.load(f)
except FileNotFoundError:
    logger.warning("ground_truth.json not found, evaluation metrics will not be computed")
    GROUND_TRUTH = []

# Cumulative evaluation counts
EVALUATION_COUNTS = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

# === Neo4j Helper Functions ===
def store_case_data(tx, case_data):
    case_id = str(uuid.uuid4())
    query = """
    CREATE (c:Case {case_id: $case_id, title: $title, judge: $judge, date: $date, case_type: $case_type, parties: $parties})
    CREATE (s:Summary {summary_id: $summary_id, text: $summary_text, created_at: $created_at})
    CREATE (v:Verdict {verdict_id: $verdict_id, text: $verdict_text, created_at: $created_at})
    CREATE (i:Issues {issues_id: $issues_id, text: $issues_text, created_at: $created_at})
    CREATE (e:Entities {entities_id: $entities_id, text: $entities_text, created_at: $created_at})
    CREATE (d:Damages {damages_id: $damages_id, text: $damages_text, amount: $damages_amount, created_at: $created_at})
    CREATE (s)-[:BELONGS_TO]->(c)
    CREATE (v)-[:BELONGS_TO]->(c)
    CREATE (i)-[:BELONGS_TO]->(c)
    CREATE (e)-[:BELONGS_TO]->(c)
    CREATE (d)-[:BELONGS_TO]->(c)
    """
    created_at = time.strftime("%Y-%m-%d")
    damages_amount = case_data.get("damages_amount", "Unknown")
    if damages_amount == "Unknown":
        damages_amount = None  # Store as null in Neo4j if not provided
    tx.run(
        query,
        case_id=case_id,
        title=case_data.get("title", "Unknown"),
        judge=case_data.get("judge", "Unknown"),
        date=case_data.get("date", created_at),
        case_type=case_data.get("case_type", "Unknown"),
        parties=case_data.get("parties", "Unknown"),
        summary_id=str(uuid.uuid4()),
        summary_text=case_data.get("summary", ""),
        verdict_id=str(uuid.uuid4()),
        verdict_text=case_data.get("verdict", ""),
        issues_id=str(uuid.uuid4()),
        issues_text=case_data.get("issues", ""),
        entities_id=str(uuid.uuid4()),
        entities_text=case_data.get("entities", ""),
        damages_id=str(uuid.uuid4()),
        damages_text=case_data.get("damages", ""),
        damages_amount=damages_amount,
        created_at=created_at,
    )
    return case_id

def query_cases(tx, filters):
    query = "MATCH (c:Case) WHERE 1=1"
    params = {}
    if filters.get("case_id"):
        query += " AND c.case_id = $case_id"
        params["case_id"] = filters["case_id"]
    if filters.get("title"):
        query += " AND toLower(c.title) CONTAINS toLower($title)"
        params["title"] = filters["title"]
    if filters.get("judge"):
        query += " AND toLower(c.judge) CONTAINS toLower($judge)"
        params["judge"] = filters["judge"]
    if filters.get("start_date") and filters.get("end_date"):
        query += " AND c.date >= $start_date AND c.date <= $end_date"
        params["start_date"] = filters["start_date"]
        params["end_date"] = filters["end_date"]
    if filters.get("case_type"):
        query += " AND c.case_type = $case_type"
        params["case_type"] = filters["case_type"]
    if filters.get("parties"):
        query += " AND toLower(c.parties) CONTAINS toLower($parties)"
        params["parties"] = filters["parties"]
    query += """
    OPTIONAL MATCH (s:Summary)-[:BELONGS_TO]->(c)
    RETURN c.case_id AS case_id, c.title AS title, c.judge AS judge, c.date AS date,
           c.case_type AS case_type, c.parties AS parties, s.text AS summary
    """
    result = tx.run(query, **params)
    return [
        {
            "case_id": record["case_id"],
            "title": record["title"],
            "judge": record["judge"],
            "date": record["date"],
            "case_type": record["case_type"],
            "parties": record["parties"],
            "summary": record["summary"],
        }
        for record in result
    ]

def query_case_details(tx, query_text):
    query = """
    MATCH (c:Case)
    OPTIONAL MATCH (s:Summary)-[:BELONGS_TO]->(c)
    OPTIONAL MATCH (v:Verdict)-[:BELONGS_TO]->(c)
    OPTIONAL MATCH (i:Issues)-[:BELONGS_TO]->(c)
    OPTIONAL MATCH (e:Entities)-[:BELONGS_TO]->(c)
    OPTIONAL MATCH (d:Damages)-[:BELONGS_TO]->(c)
    WHERE toLower(c.title) CONTAINS toLower($query_text)
       OR toLower(s.text) CONTAINS toLower($query_text)
       OR toLower(v.text) CONTAINS toLower($query_text)
       OR toLower(i.text) CONTAINS toLower($query_text)
       OR toLower(e.text) CONTAINS toLower($query_text)
       OR toLower(d.text) CONTAINS toLower($query_text)
    RETURN c.case_id AS case_id, c.title AS title, c.date AS date, s.text AS summary,
           v.text AS verdict, i.text AS issues, e.text AS entities,
           c.judge AS judge, d.text AS damages, d.amount AS damages_amount
    LIMIT 5
    """
    result = tx.run(query, query_text=query_text)
    return [
        {
            "case_id": record["case_id"],
            "title": record["title"] if record["title"] else "Unknown",
            "date": record["date"] if record["date"] else "Not specified",
            "summary": record["summary"] if record["summary"] else "Not available",
            "verdict": record["verdict"] if record["verdict"] else "Not specified",
            "issues": record["issues"] if record["issues"] else "Not specified",
            "entities": record["entities"] if record["entities"] else "Not specified",
            "judge": record["judge"] if record["judge"] else "Not specified",
            "damages": record["damages"] if record["damages"] else "Not specified",
            "damages_amount": record["damages_amount"] if record["damages_amount"] else "Not specified",
        }
        for record in result
    ]

# === Routes ===
@app.route('/')
def home():
    return "Legal Case Analysis Chatbot API is up. Use /upload, /search, or /chat."

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        logger.error("No file provided in request")
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if not file.filename.lower().endswith(('.pdf', '.docx')):
        logger.error(f"Unsupported file type: {file.filename}")
        return jsonify({'error': 'Unsupported file type. Only PDF and DOCX are allowed.'}), 400

    # Validate file size
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    if file_size > MAX_FILE_SIZE:
        logger.error(f"File too large: {file_size} bytes")
        return jsonify({'error': f'File too large. Max size is {MAX_FILE_SIZE // (1024 * 1024)}MB.'}), 400
    file.seek(0)

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    try:
        file.save(filepath)
        logger.info(f"File saved: {filepath}")
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        return jsonify({'error': f'Error saving file: {str(e)}'}), 500

    try:
        text = extract_text(filepath)
        logger.info(f"Extracted text length: {len(text)} characters")
        chunks = chunk_text(text)

        # Summarize and extract case details using Groq API
        case_data = extract_case_data(chunks)
        logger.info(f"Extracted case data: {case_data}")

        # Store in Neo4j
        with neo4j_driver.session() as session:
            case_id = session.execute_write(store_case_data, case_data)
            logger.info(f"Case stored in Neo4j with case_id: {case_id}")

        return jsonify({
            'message': 'File processed successfully!',
            'case_id': case_id,
            'filename': file.filename,
            'summary': case_data.get('summary', 'No summary available.')
        }), 200
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/search', methods=['POST'])
def search_cases():
    try:
        filters = request.get_json()
        logger.info(f"Search filters: {filters}")
        with neo4j_driver.session() as session:
            results = session.execute_read(query_cases, filters)
        logger.info(f"Search results: {results}")
        return jsonify({'cases': results}), 200
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return jsonify({'error': f'Search error: {str(e)}'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get('message')
        case_id = data.get('case_id')  # Get the case_id from the frontend
        logger.info(f"Chat request - Message: {message}, Case ID: {case_id}")

        if not message:
            logger.error("No message provided in chat request")
            return jsonify({'error': 'No message provided'}), 400
        if not case_id:
            logger.error("No case_id provided in chat request")
            return jsonify({'error': 'No case ID provided'}), 400

        # Query Neo4j for relevant case data, prioritizing the provided case_id
        with neo4j_driver.session() as session:
            query = """
            MATCH (c:Case {case_id: $case_id})
            OPTIONAL MATCH (s:Summary)-[:BELONGS_TO]->(c)
            OPTIONAL MATCH (v:Verdict)-[:BELONGS_TO]->(c)
            OPTIONAL MATCH (i:Issues)-[:BELONGS_TO]->(c)
            OPTIONAL MATCH (e:Entities)-[:BELONGS_TO]->(c)
            OPTIONAL MATCH (d:Damages)-[:BELONGS_TO]->(c)
            RETURN c.case_id AS case_id, c.title AS title, c.date AS date, s.text AS summary,
                   v.text AS verdict, i.text AS issues, e.text AS entities,
                   c.judge AS judge, d.text AS damages, d.amount AS damages_amount
            """
            try:
                result = session.run(query, case_id=case_id)
                case_details = [
                    {
                        "case_id": record["case_id"],
                        "title": record["title"] if record["title"] else "Unknown",
                        "date": record["date"] if record["date"] else "Not specified",
                        "summary": record["summary"] if record["summary"] else "Not available",
                        "verdict": record["verdict"] if record["verdict"] else "Not specified",
                        "issues": record["issues"] if record["issues"] else "Not specified",
                        "entities": record["entities"] if record["entities"] else "Not specified",
                        "judge": record["judge"] if record["judge"] else "Not specified",
                        "damages": record["damages"] if record["damages"] else "Not specified",
                        "damages_amount": record["damages_amount"] if record["damages_amount"] else "Not specified",
                    }
                    for record in result
                ]
                logger.info(f"Case details retrieved for case_id {case_id}: {case_details}")
            except Exception as e:
                logger.error(f"Neo4j query error in /chat: {str(e)}")
                case_details = []
                return jsonify({'error': f'Failed to retrieve case data from Neo4j: {str(e)}'}), 500

            # If no case found, fall back to broader search
            if not case_details:
                logger.warning(f"No case found with case_id {case_id}, attempting to search by query text")
                try:
                    case_details = session.execute_read(query_case_details, message)
                    logger.info(f"Fallback search results: {case_details}")
                except Exception as e:
                    logger.error(f"Neo4j fallback search error: {str(e)}")
                    return jsonify({'error': f'Failed to search case data in Neo4j: {str(e)}'}), 500

        if case_details:
            # Validate and clean case details
            for case in case_details:
                for key in ["title", "judge", "date", "verdict", "issues", "entities", "damages"]:
                    if not case[key] or case[key] in ["Unknown", "Not specified", "Not available", None]:
                        logger.warning(f"Field '{key}' missing or invalid for case {case['case_id']}: {case[key]}")
                        case[key] = "Not specified in the document."
                if not case["damages_amount"] or case["damages_amount"] in ["Unknown", "Not specified", None]:
                    case["damages_amount"] = "Not specified in the document."

            # Build context for the Groq API, including damages and date
            context = "\n".join(
                [
                    f"Case: {d['title']}\nDate: {d['date']}\nJudge: {d.get('judge', 'Not specified')}\nVerdict: {d['verdict']}\nIssues: {d['issues']}\nEntities: {d['entities']}\nDamages: {d['damages']}\nDamages Amount: {d['damages_amount']}"
                    for d in case_details
                ]
            )
            logger.info(f"Context for chatbot: {context}")
            prompt = f"Based on the following case data:\n{context}\n\nAnswer the question: {message}\nProvide a clear and accurate response related to the case data. Do not include the summary in your response. If the question asks about damages or dates, use the specific details provided."
        else:
            logger.warning("No relevant case data found, using fallback prompt")
            prompt = f"No relevant case data found for the provided case ID or query. Answer the question to the best of your knowledge: {message}\nProvide a clear and accurate response, or indicate if the information is not available."

        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a legal expert assistant. Provide accurate and relevant answers based on the given case data, especially for dates and damages."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=500,
            )
            reply = response.choices[0].message.content.strip()
            logger.info(f"Chat response: {reply}")
        except Exception as e:
            logger.error(f"Groq API error in /chat: {str(e)}")
            reply = "I’m sorry, I couldn’t process your request due to an issue with the AI service. Please try again later or upload the document again."

        # Evaluate response against ground truth
        evaluation = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for gt in GROUND_TRUTH:
            if gt["question"].lower() == message.lower() and gt["case_id"] == case_id:
                # Use string similarity to compare answers (threshold: 0.85)
                similarity = SequenceMatcher(None, reply.lower(), gt["expected_answer"].lower()).ratio()
                if similarity >= 0.85:
                    evaluation["tp"] += 1
                    logger.info(f"Evaluation: True Positive for question '{message}' (similarity: {similarity:.2f})")
                else:
                    evaluation["fp"] += 1
                    logger.warning(f"Evaluation: False Positive for question '{message}' (similarity: {similarity:.2f}, expected: {gt['expected_answer']}, got: {reply})")
                break
        else:
            # Question not in ground truth or wrong case_id
            if any(gt["question"].lower() == message.lower() for gt in GROUND_TRUTH):
                evaluation["fn"] += 1
                logger.warning(f"Evaluation: False Negative for question '{message}' (not answered correctly for case_id {case_id})")
            else:
                evaluation["tn"] += 1
                logger.info(f"Evaluation: True Negative for question '{message}' (not in ground truth)")

        # Calculate per-question metrics
        total = evaluation["tp"] + evaluation["fp"] + evaluation["fn"] + evaluation["tn"]
        accuracy = (evaluation["tp"] + evaluation["tn"]) / total if total > 0 else 0
        precision = evaluation["tp"] / (evaluation["tp"] + evaluation["fp"]) if (evaluation["tp"] + evaluation["fp"]) > 0 else 0
        recall = evaluation["tp"] / (evaluation["tp"] + evaluation["fn"]) if (evaluation["tp"] + evaluation["fn"]) > 0 else 0

        logger.info(f"Evaluation Metrics: Accuracy={accuracy:.2f}, Precision={precision:.2f}, Recall={recall:.2f}, TP={evaluation['tp']}, FP={evaluation['fp']}, FN={evaluation['fn']}, TN={evaluation['tn']}")

        # Update cumulative evaluation counts
        for key in evaluation:
            EVALUATION_COUNTS[key] += evaluation[key]
        
        # Calculate overall accuracy
        total_cumulative = EVALUATION_COUNTS["tp"] + EVALUATION_COUNTS["fp"] + EVALUATION_COUNTS["fn"] + EVALUATION_COUNTS["tn"]
        overall_accuracy = (EVALUATION_COUNTS["tp"] + EVALUATION_COUNTS["tn"]) / total_cumulative if total_cumulative > 0 else 0
        logger.info(f"Overall Accuracy={overall_accuracy*100:.2f}% (TP={EVALUATION_COUNTS['tp']}, FP={EVALUATION_COUNTS['fp']}, FN={EVALUATION_COUNTS['fn']}, TN={EVALUATION_COUNTS['tn']})")

        response_data = {
            'reply': reply,
            'evaluation': {
                'accuracy': round(accuracy, 2),
                'precision': round(precision, 2),
                'recall': round(recall, 2),
                'tp': evaluation['tp'],
                'fp': evaluation['fp'],
                'fn': evaluation['fn'],
                'tn': evaluation['tn'],
                'overall_accuracy': round(overall_accuracy, 2)
            }
        }
        logger.info(f"Chat endpoint response: {response_data}")
        return jsonify(response_data), 200

    except Exception as e:
        logger.error(f"Unexpected error in /chat endpoint: {str(e)}")
        return jsonify({'error': f'Unexpected chat error: {str(e)}'}), 500

# === Helper Functions ===
def extract_text(filepath):
    try:
        ext = filepath.split('.')[-1].lower()
        logger.info(f"Extracting text from file: {filepath} (type: {ext})")
        text = ''
        if ext == 'pdf':
            doc = fitz.open(filepath)
            for page in doc:
                page_text = page.get_text()
                if page_text:
                    text += page_text + "\n"
            doc.close()
            logger.info(f"Extracted {len(text)} characters from PDF")
        elif ext == 'docx':
            doc = docx.Document(filepath)
            text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            logger.info(f"Extracted {len(text)} characters from DOCX")
        else:
            logger.error("Unsupported file type")
            raise ValueError("Unsupported file type.")
        if not text.strip():
            logger.error("Extracted text is empty after processing")
            raise ValueError("No readable text found in the document.")
        return text
    except Exception as e:
        logger.error(f"Text extraction error: {str(e)}")
        raise ValueError(f"Error extracting text from document: {str(e)}")

def chunk_text(text, size=500):
    words = text.split()
    chunks = [' '.join(words[i:i + size]) for i in range(0, len(words), size)]
    logger.info(f"Text chunked into {len(chunks)} chunks")
    return chunks

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def extract_case_data(chunks):
    combined_text = " ".join(chunks)
    logger.info(f"Combined text length for Groq API: {len(combined_text)} characters")
    prompt = """
    You are a legal expert assistant. Extract the following from the legal case text and return the result as a valid JSON object. Ensure the response is strictly in JSON format, with all fields properly enclosed in double quotes, and do not include any additional text outside the JSON object. The fields to extract are:
    - "title": Title of the case (e.g., "Smith vs. Jones")
    - "judge": Judge name (e.g., "Judge Smith")
    - "date": Date of the case in YYYY-MM-DD format (e.g., "2023-05-01")
    - "case_type": Case type (e.g., "Criminal", "Civil")
    - "parties": Parties involved (e.g., "John Smith and Jane Jones")
    - "summary": Summary of the case (a paragraph of 4-5 sentences in simple, easy-to-understand language)
    - "verdict": Verdict (e.g., "In favor of the plaintiff")
    - "issues": Legal issues and citations (e.g., "Breach of contract, Section 5")
    - "entities": Key entities and relationships (e.g., "John Smith (Plaintiff), Jane Jones (Defendant)")
    - "damages": Details of damages awarded (e.g., "Total $500,000, including $300,000 for costs and $200,000 for penalties")
    - "damages_amount": Total damages amount as a string (e.g., "$500,000")
    Ensure the summary is a complete paragraph with proper punctuation. Ensure the date is in YYYY-MM-DD format, parsing text like 'March 15, 2024' correctly. For damages, include both a detailed description and the total amount. If any field cannot be determined, use "Unknown" as the value, but ensure the "summary" field is at least a basic summary of the text provided. Log any issues with date or damages extraction.
    """
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": combined_text},
            ],
            temperature=0.3,
            max_tokens=1000,
        )
        raw_response = response.choices[0].message.content.strip()
        logger.info(f"Groq API raw response in extract_case_data: {raw_response}")

        # Validate that the response is proper JSON
        try:
            case_data = json.loads(raw_response)
            # Ensure all required fields are present
            required_fields = ["title", "judge", "date", "case_type", "parties", "summary", "verdict", "issues", "entities", "damages", "damages_amount"]
            for field in required_fields:
                if field not in case_data:
                    logger.warning(f"Missing field '{field}' in Groq API response, setting to 'Unknown'")
                    case_data[field] = "Unknown"
                if not case_data[field]:
                    logger.warning(f"Empty field '{field}' in Groq API response, setting to 'Unknown'")
                    case_data[field] = "Unknown"
            # Validate date format (YYYY-MM-DD)
            if case_data["date"] != "Unknown":
            # Validate date format (YYYY-MM-DD)
                try:
                    time.strptime(case_data["date"], "%Y-%m-%d")
                except ValueError:
                    logger.warning(f"Invalid date format in Groq response: {case_data['date']}, setting to 'Unknown'")
                    case_data["date"] = "Unknown"
            # Ensure damages_amount is a string
            if case_data["damages_amount"] != "Unknown" and not isinstance(case_data["damages_amount"], str):
                logger.warning(f"Invalid damages_amount type: {type(case_data['damages_amount'])}, converting to string")
                case_data["damages_amount"] = str(case_data["damages_amount"])
            # Ensure summary is meaningful
            if "summary" in case_data and "Summary truncated" in case_data["summary"]:
                case_data["summary"] = combined_text[:500] + "... Summary extraction failed."
            return case_data
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in extract_case_data: {str(e)}. Raw response: {raw_response}")
            return {
                "title": "Unknown",
                "judge": "Unknown",
                "date": time.strftime("%Y-%m-%d"),
                "case_type": "Unknown",
                "parties": "Unknown",
                "summary": combined_text[:500] + "... Summary extraction failed due to invalid JSON response.",
                "verdict": "Unknown",
                "issues": "Unknown",
                "entities": "Unknown",
                "damages": "Unknown",
                "damages_amount": "Unknown",
            }
    except Exception as e:
        logger.error(f"Groq API error in extract_case_data: {str(e)}")
        return {
            "title": "Unknown",
            "judge": "Unknown",
            "date": time.strftime("%Y-%m-%d"),
            "case_type": "Unknown",
            "parties": "Unknown",
            "summary": combined_text[:500] + "... Summary extraction failed due to API error.",
            "verdict": "Unknown",
            "issues": "Unknown",
            "entities": "Unknown",
            "damages": "Unknown",
            "damages_amount": "Unknown",
        }

if __name__ == '__main__':
    try:
        app.run(debug=True, port=5001)
    finally:
        neo4j_driver.close()