A Flask-based chatbot for analyzing legal case documents, extracting key details (e.g., damages, dates, verdicts), and answering user queries. The backend uses Neo4j for data storage, Groq API for text processing, and logs performance metrics (accuracy, precision, recall, overall accuracy) to evaluate responses against a ground truth dataset. The frontend is built with React for an interactive user interface.

Features





Document Upload: Upload PDF or DOCX legal case documents (e.g., Smith Enterprises, Inc. v. Johnson Construction, LLC).



Data Extraction: Extracts case details (title, judge, date, damages, verdict, etc.) using Groq API.



Query Handling: Answers questions about case details, prioritizing damages and dates.



Evaluation Metrics: Logs accuracy, precision, recall, and overall accuracy for responses, compared to ground_truth.json.



Logging: Saves detailed logs to logs/legal_chatbot.log with rotating file handler (5MB, 5 backups).

Prerequisites





Python 3.8+



Node.js 14+ (for React frontend)



Neo4j Database (local or cloud, e.g., neo4j://localhost:7687)



Groq API Key (from xAI)
