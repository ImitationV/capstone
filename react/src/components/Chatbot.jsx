import React, { useState, useRef, useEffect } from 'react';
import '../styles/chatbot.css';
import chatbotIcon from '../assets/chatbot.png';

function Chatbot() {
  const [isOpen, setIsOpen] = useState(false);
  const [userInput, setUserInput] = useState('');
  const [messages, setMessages] = useState([
    { text: 'Hello, how are you doing? How can I help you today?', sender: 'assistant' },
  ]);

  const chatMessagesRef = useRef(null); // Ref to chat messages div

  useEffect(() => {
    if (chatMessagesRef.current) {
      chatMessagesRef.current.scrollTop = chatMessagesRef.current.scrollHeight;
    }
  }, [messages]);

  useEffect(() => {
    console.log('Current Messages:', messages);
  }, [messages]);

  const toggleChat = () => {
    setIsOpen(!isOpen);
  };

  const handleInputChange = (event) => {
    setUserInput(event.target.value);
  };

  const handleSendMessage = () => {
    if (userInput.trim() !== '') {
      setMessages([...messages, { text: userInput, sender: 'user' }]);
      setUserInput('');
    }
  };

  return ( // Add 'open' class if isOpen is true
    <div className={`chatbot-container ${isOpen ? 'open' : ''}`}>  
      <div className="chatbot-icon" onClick={toggleChat}>
        <img src={chatbotIcon} alt="Chatbot Icon" style={{ width: '50%', height: '50%', borderRadius: '10%' }} /> 
      </div>
      {isOpen && (
        <div className="chatbot-window">
          <div className="chatbot-header">
            <div className="logo">Budget Buddy Chat</div>
            <div className="close-button" onClick={toggleChat}>
              X
            </div>
          </div>
          <div className="chat-messages" ref={chatMessagesRef}>
            {messages.map((message, index) => (
              <div
                key={index}
                className={`message ${message.sender === 'assistant' ? 'assistant' : 'user'}`}
              >
                {message.text}
              </div>
            ))}
          </div>
          <div className="chat-input">
            <input
              type="text"
              placeholder="Reply..."
              value={userInput}
              onChange={handleInputChange}
            />
            <button onClick={handleSendMessage}>&gt;</button>
          </div>
        </div>
      )}
    </div>
  );
}

export default Chatbot;