import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [file, setFile] = useState(null);
  const [summary, setSummary] = useState('');
  const [error, setError] = useState('');
  const [userMessage, setUserMessage] = useState('');
  const [chatLog, setChatLog] = useState([]);
  const [documentId, setDocumentId] = useState('');
  const [filename, setFilename] = useState('');

  // Handle file selection
  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setError('');
  };

  // Handle file upload
  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file first.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      const uploadRes = await axios.post('http://localhost:5001/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      setSummary(uploadRes.data.summary || 'No summary available.');
      setDocumentId(uploadRes.data.case_id);
      setFilename(uploadRes.data.filename || file.name);
      setChatLog([]);
      setError('');
    } catch (err) {
      console.error('Upload error:', err);
      setError(err.response?.data?.error || 'Something went wrong during upload.');
    }
  };

  // Handle sending chat messages
  const handleSendMessage = async () => {
    if (!userMessage.trim()) {
      setError('Please enter a message.');
      return;
    }

    const newChatLog = [...chatLog, { role: 'user', content: userMessage }];
    setChatLog(newChatLog);
    setUserMessage('');
    setError('');

    try {
      const res = await axios.post('http://localhost:5001/chat', {
        message: userMessage,
        case_id: documentId,  // Send the case_id to prioritize the recently uploaded document
      });
      const reply = res.data.reply;

      setChatLog([
        ...newChatLog,
        {
          role: 'assistant',
          content: reply // Only include the reply, remove cases_used
        }
      ]);
    } catch (err) {
      console.error('Chat error:', err.response?.data, err.message);
      const errorMsg = err.response?.data?.error || 'Something went wrong. Try again.';
      setChatLog([...newChatLog, { role: 'assistant', content: errorMsg }]);
      setError(errorMsg);
    }
  };

  return (
    <div
      style={{
        padding: '30px',
        maxWidth: '900px',
        margin: '0 auto',
        fontFamily: "'Roboto', sans-serif",
        background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)',
        minHeight: '100vh',
        color: '#e0e0e0',
      }}
    >
      <h1
        style={{
          fontSize: '32px',
          marginBottom: '30px',
          color: '#ff00ff',
          textShadow: '0 0 10px #ff00ff, 0 0 20px #ff00ff, 0 0 30px #ff00ff',
          textAlign: 'center',
        }}
      >
        📁 Legal Document Summarizer
      </h1>

      {/* File Upload Section */}
      <div
        style={{
          marginBottom: '30px',
          padding: '20px',
          background: 'rgba(255, 255, 255, 0.05)',
          borderRadius: '10px',
          border: '1px solid #00ffff',
          boxShadow: '0 0 15px #00ffff',
        }}
      >
        <input
          type="file"
          accept=".pdf,.docx"
          onChange={handleFileChange}
          style={{
            marginRight: '15px',
            padding: '10px',
            background: 'rgba(255, 255, 255, 0.1)',
            border: '1px solid #ff00ff',
            borderRadius: '5px',
            color: '#e0e0e0',
            cursor: 'pointer',
            boxShadow: '0 0 5px #ff00ff',
          }}
        />
        <button
          onClick={handleUpload}
          style={{
            padding: '10px 20px',
            background: '#ff00ff',
            color: '#fff',
            border: 'none',
            borderRadius: '5px',
            cursor: 'pointer',
            boxShadow: '0 0 10px #ff00ff',
            transition: 'all 0.3s ease',
          }}
          onMouseOver={(e) => (e.target.style.boxShadow = '0 0 20px #ff00ff')}
          onMouseOut={(e) => (e.target.style.boxShadow = '0 0 10px #ff00ff')}
        >
          Upload & Summarize
        </button>
        {error && (
          <p
            style={{
              color: '#ff5555',
              marginTop: '15px',
              fontSize: '14px',
              textShadow: '0 0 5px #ff5555',
            }}
          >
            {error}
          </p>
        )}
      </div>

      {/* Summary Section */}
      {summary && (
        <div
          style={{
            marginBottom: '30px',
            padding: '20px',
            background: 'rgba(255, 255, 255, 0.05)',
            borderRadius: '10px',
            border: '1px solid #00ff00',
            boxShadow: '0 0 15px #00ff00',
          }}
        >
          <h2
            style={{
              fontSize: '24px',
              marginBottom: '15px',
              color: '#00ffff',
              textShadow: '0 0 5px #00ffff',
            }}
          >
            📝 Case Summary
          </h2>
          <p
            style={{
              border: '1px solid #ff00ff',
              padding: '15px',
              borderRadius: '5px',
              background: 'rgba(255, 255, 255, 0.1)',
              whiteSpace: 'pre-wrap',
              color: '#e0e0e0',
              boxShadow: '0 0 5px #ff00ff',
              fontSize: '14px',
            }}
          >
            {summary}
          </p>
        </div>
      )}

      {/* Chat Section */}
      <div
        style={{
          padding: '20px',
          background: 'rgba(255, 255, 255, 0.05)',
          borderRadius: '10px',
          border: '1px solid #00ffff',
          boxShadow: '0 0 15px #00ffff',
        }}
      >
        <h2
          style={{
            fontSize: '24px',
            marginBottom: '15px',
            color: '#00ff00',
            textShadow: '0 0 5px #00ff00',
          }}
        >
          💬 Chat About the Case
        </h2>
        <div
          style={{
            border: '1px solid #ff00ff',
            padding: '20px',
            minHeight: '250px',
            marginBottom: '15px',
            borderRadius: '5px',
            background: 'rgba(255, 255, 255, 0.1)',
            overflowY: 'auto',
            boxShadow: '0 0 10px #ff00ff',
          }}
        >
          {chatLog.length === 0 ? (
            <p
              style={{
                color: '#888',
                fontStyle: 'italic',
                textAlign: 'center',
              }}
            >
              Start by asking a question about the case...
            </p>
          ) : (
            chatLog.map((msg, i) => (
              <p
                key={i}
                style={{
                  margin: '10px 0',
                  padding: '10px',
                  background:
                    msg.role === 'user'
                      ? 'rgba(0, 255, 255, 0.2)'
                      : 'rgba(255, 0, 255, 0.2)',
                  borderRadius: '5px',
                  fontSize: '14px',
                  boxShadow:
                    msg.role === 'user'
                      ? '0 0 5px #00ffff'
                      : '0 0 5px #ff00ff',
                }}
              >
                <strong
                  style={{
                    color: msg.role === 'user' ? '#00ffff' : '#ff00ff',
                  }}
                >
                  {msg.role === 'user' ? 'You' : 'AI'}:
                </strong>{' '}
                {msg.content}
              </p>
            ))
          )}
        </div>
        <div style={{ display: 'flex', gap: '15px' }}>
          <input
            style={{
              flex: 1,
              padding: '10px',
              border: '1px solid #00ff00',
              borderRadius: '5px',
              background: 'rgba(255, 255, 255, 0.1)',
              color: '#e0e0e0',
              fontSize: '14px',
              boxShadow: '0 0 5px #00ff00',
            }}
            placeholder="Ask a question about the document..."
            value={userMessage}
            onChange={(e) => setUserMessage(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
          />
          <button
            onClick={handleSendMessage}
            style={{
              padding: '10px 20px',
              background: '#00ffff',
              color: '#fff',
              border: 'none',
              borderRadius: '5px',
              cursor: 'pointer',
              boxShadow: '0 0 10px #00ffff',
              transition: 'all 0.3s ease',
            }}
            onMouseOver={(e) => (e.target.style.boxShadow = '0 0 20px #00ffff')}
            onMouseOut={(e) => (e.target.style.boxShadow = '0 0 10px #00ffff')}
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;