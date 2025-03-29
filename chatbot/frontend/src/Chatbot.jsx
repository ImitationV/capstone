import React, { useState, useEffect } from 'react';

export default function Chatbot() {
  const [messages, setMessages] = useState([
    { sender: "bot", text: "Hello! I'm your financial assistant. How can I help you today?" }
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [backendStatus, setBackendStatus] = useState("checking");
  
  // Your Hugging Face Space URL (note the correct format)
  const API_BASE = "https://mahdee987-financial-chatbot.hf.space";

  // Check backend status on component mount
  useEffect(() => {
    const checkStatus = async () => {
      try {
        const response = await fetch(`${API_BASE}/`);
        if (response.ok) {
          const data = await response.json();
          console.log("Backend status:", data);
          setBackendStatus("online");
        } else {
          setBackendStatus("offline");
        }
      } catch (error) {
        console.error("Status check failed:", error);
        setBackendStatus("offline");
      }
    };
    
    checkStatus();
  }, []);

  const sendMessage = async () => {
    if (!input.trim() || backendStatus !== "online") return;

    const userMessage = { sender: "user", text: input };
    setMessages(prev => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { 
          "Content-Type": "application/json",
          "Accept": "application/json"
        },
        body: JSON.stringify({ message: input })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setMessages(prev => [...prev, { 
        sender: "bot", 
        text: data.response || "I didn't get a proper response."
      }]);
    } catch (error) {
      console.error("API Error:", error);
      setMessages(prev => [...prev, { 
        sender: "bot", 
        text: "Sorry, I'm having trouble connecting. Please try again later."
      }]);
      
      // Re-check backend status after failure
      setBackendStatus("checking");
      setTimeout(() => {
        fetch(`${API_BASE}/`)
          .then(res => setBackendStatus(res.ok ? "online" : "offline"))
          .catch(() => setBackendStatus("offline"));
      }, 5000);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container mt-5">
      <div className="card shadow-lg rounded">
        {/* Enhanced Header with Status Indicator */}
        <div className="card-header bg-primary text-white d-flex justify-content-between align-items-center">
          <div className="d-flex align-items-center">
            <div className="rounded-circle bg-white text-dark d-flex align-items-center justify-content-center me-2" 
                 style={{ width: "40px", height: "40px", fontWeight: "bold" }}>
              BB
            </div>
            <div>
              <h5 className="mb-0">Financial Assistant</h5>
              <small className="d-flex align-items-center">
                Status: 
                <span className={`ms-2 badge ${backendStatus === "online" ? "bg-success" : 
                                 backendStatus === "offline" ? "bg-danger" : "bg-warning"}`}>
                  {backendStatus === "checking" ? (
                    <>
                      <span className="spinner-border spinner-border-sm me-1" role="status"></span>
                      Checking...
                    </>
                  ) : (
                    backendStatus.toUpperCase()
                  )}
                </span>
              </small>
            </div>
          </div>
          <button className="btn btn-light btn-sm">&times;</button>
        </div>

        {/* Chat Messages */}
        <div className="card-body overflow-auto" style={{ height: "400px" }}>
          {messages.map((msg, index) => (
            <div key={index} className={`mb-2 d-flex ${msg.sender === "user" ? "justify-content-end" : "justify-content-start"}`}>
              <div className={`p-3 rounded ${msg.sender === "user" ? "bg-primary text-white" : "bg-light text-dark"}`} 
                   style={{ maxWidth: "75%", wordWrap: "break-word" }}>
                {msg.text}
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="d-flex justify-content-start">
              <div className="p-3 rounded bg-light text-dark" style={{ maxWidth: "75%" }}>
                <div className="spinner-border spinner-border-sm text-secondary me-2" role="status"></div>
                Processing your question...
              </div>
            </div>
          )}
        </div>

        {/* Enhanced Input with Status Awareness */}
        <div className="card-footer bg-light d-flex">
          <input
            type="text"
            className="form-control me-2"
            placeholder={backendStatus === "online" ? "Ask about personal finance..." : "Service unavailable..."}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === "Enter" && sendMessage()}
            disabled={isLoading || backendStatus !== "online"}
          />
          <button 
            className="btn btn-primary" 
            onClick={sendMessage}
            disabled={isLoading || !input.trim() || backendStatus !== "online"}
            style={{ minWidth: "40px" }}
          >
            {isLoading ? (
              <span className="spinner-border spinner-border-sm" role="status"></span>
            ) : (
              "➤"
            )}
          </button>
        </div>
      </div>
      
      {/* Debug Info (remove in production) */}
      <div className="mt-3 text-muted small">
        <p>Backend Info: GPT-2 on CPU | Torch v2.0.1 | {backendStatus}</p>
      </div>
    </div>
  );
}