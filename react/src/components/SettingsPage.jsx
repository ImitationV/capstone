import React from 'react';
import '../styles/overview.css';

const SettingsPage = () => (
  <div className="settings-page" style={{ maxWidth: 600, margin: '2rem auto', textAlign: 'center' }}>
    <h1 style={{ marginBottom: '1.5rem', color: '#2d3748' }}>More Coming Soon</h1>
    <div style={{ background: '#f9fafb', borderRadius: 8, padding: '1.5rem', boxShadow: '0 2px 8px #e2e8f0' }}>
      <h2 style={{ color: '#4a5568', marginBottom: '0.5rem' }}>Capstone Project Disclaimer</h2>
      <p style={{ color: '#555', fontSize: '1.1rem' }}>
        This app was created as part of our capstone project using the following technologies:
        <ul>
          <li>React</li>
          <li>Node.js</li>
          <li>Express</li>
          <li>Supabase</li>
          <li>Python</li>
          <li>Flask</li>
          <li>Pandas</li>
          <li>Scikit-learn</li>
          <li>Matplotlib</li>
          <li>Seaborn</li>
          <li>Google Gemini API</li>
          <li>yFinance API</li>
          </ul>
          
      </p>
    </div>
  </div>
);

export default SettingsPage; 