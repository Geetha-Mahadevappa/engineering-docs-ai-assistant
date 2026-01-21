import { useState } from "react";
import MessageBubble from "./MessageBubble.jsx";
import SourceList from "./SourceList.jsx";

export default function Chat() {
  const [sessionId] = useState(() => crypto.randomUUID());
  const [message, setMessage] = useState("");
  const [chat, setChat] = useState([
    {
      role: "assistant",
      text: "Hi! Upload a document or ask anything about your engineering docs.",
      sources: []
    }
  ]);

  async function sendMessage() {
    if (!message.trim()) return;

    // Add user message immediately
    setChat(prev => [...prev, { role: "user", text: message }]);

    const query = `
      mutation Chat($sessionId: String!, $message: String!) {
        chat(sessionId: $sessionId, message: $message) {
          reply
          sources {
            id
            score
            text
          }
        }
      }
    `;

    try {
      const res = await fetch("http://localhost:8000/graphql", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          variables: { sessionId, message }
        })
      });

      const json = await res.json();

      if (!json.data || !json.data.chat) {
        throw new Error("Invalid backend response");
      }

      const reply = json.data.chat.reply;
      const sources = json.data.chat.sources;

      setChat(prev => [
        ...prev,
        { role: "assistant", text: reply, sources }
      ]);
    } catch (err) {
      console.error(err);
      setChat(prev => [
        ...prev,
        {
          role: "assistant",
          text: "Backend is not responding. Make sure the FastAPI server is running on port 8000.",
          sources: []
        }
      ]);
    }

    setMessage("");
  }

  return (
    <div className="chat-container">
      <div className="messages">
        {chat.map((msg, i) => (
          <div key={i}>
            <MessageBubble role={msg.role} text={msg.text} />
            {msg.sources && msg.sources.length > 0 && (
              <SourceList sources={msg.sources} />
            )}
          </div>
        ))}
      </div>

      <div className="input-row">
        <input
          className="chat-input"
          value={message}
          onChange={e => setMessage(e.target.value)}
          placeholder="Ask something about your engineering docs..."
        />
        <button className="send-btn" onClick={sendMessage}>
          Send
        </button>
      </div>
    </div>
  );
}
