import React, { useState, useEffect, useRef } from 'react';
import '../styles/chatbotPopup.css';

const ChatbotPopup = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [userInput, setUserInput] = useState('');
  const messagesEndRef = useRef(null);

  const togglePopup = () => setIsOpen(!isOpen);

  const sendMessage = async () => {
    if (!userInput.trim()) return;
    const userMessage = { role: 'user', text: userInput };
    const thinkingMessage = { role: 'bot', text: 'Thinking...' };

    // Show user message and placeholder
    const updatedMessages = [...messages, userMessage, thinkingMessage];
    setMessages(updatedMessages);
    setUserInput('');

    try {
      const res = await fetch('http://127.0.0.1:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userInput }),
      });
      const data = await res.json();

      // Replace "Thinking..." with actual response
      const finalMessages = [
        ...messages,
        userMessage,
        { role: 'bot', text: data.response || 'No response.' },
      ];
      setMessages(finalMessages);
    } catch (err) {
      const errorMessages = [
        ...messages,
        userMessage,
        { role: 'bot', text: 'Error fetching response.' },
      ];
      setMessages(errorMessages);
    }
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div className={`chatbot-container ${isOpen ? 'open' : ''}`}>
      {isOpen ? (
        <div className="chatbot-box">
          <div className="chatbot-header">
            <span>Financial Assistant</span>
            <button onClick={togglePopup}>−</button>
          </div>
          <div className="chatbot-messages">
            {messages.map((msg, i) => (
              <div key={i} className={`chat-message ${msg.role}`}>
                {msg.text}
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
          <div className="chatbot-input">
            <input
              type="text"
              value={userInput}
              onChange={(e) => setUserInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
              placeholder="Ask something..." />
            <button onClick={sendMessage}>Send</button>
          </div>
        </div>
      ) : (
        <button className="chatbot-button" onClick={togglePopup}>
          💬
        </button>
      )}
    </div>
  );
};

export default ChatbotPopup;
