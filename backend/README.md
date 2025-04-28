# Chatbot Backend

This is a FastAPI backend powered by **LangChain**, **Google Gemini 2.0**, and **Pydantic**. It provides clear and actionable financial advice in response to user queries via a `/chat` API endpoint.

---

## Features

- 🌐 FastAPI backend with CORS enabled
- 🧠 Google Gemini 2.0 (`gemini-2.0-flash`) model via LangChain
- 🔍 Output parsing using `PydanticOutputParser`
- 🧾 Structured financial advice using Pydantic schemas
- 🌱 Environment-based configuration with `.env`

---

## dependencies
- pip install -r requirements.txt
- Create a .env file in the backend directory and add your Google API key:
- GOOGLE_API_KEY=your_google_api_key_here
- uvicorn main:app --reload (on backend)
- npm install (react/frontend)
- npm run dev/ npm start

---

## Tech Stack

- FastAPI
- LangChain
- Google Gemini 2.0 (via langchain-google-genai)
- Pydantic
- CORS Middleware
- doten

---

## License

- Feel free to use this however you want. Just give me credit, and don’t blame me if something breaks. And always remember if it works, DONT TOUCH IT!
