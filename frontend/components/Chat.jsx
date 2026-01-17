import { useState } from "react";
import MessageBubble from "./MessageBubble.jsx";
import SourceList from "./SourceList.jsx";

export default function Chat() {
  const [sessionId] = useState(() => crypto.randomUUID());
  const [message, setMessage] = useState("");
  const [chat, setChat] = useState([]);

  async function sendMessage() {
    if (!message.trim()) return;

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

    const res = await fetch("http://localhost:8000/graphql", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        variables: { sessionId, message }
      })
    });

    const json = await res.json();
    const reply = json.data.chat.reply;
    const sources = json.data.chat.sources;

    setChat(prev => [
      ...prev,
      { role: "user", text: message },
      { role: "assistant", text: reply, sources }
    ]);

    setMessage("");
  }

  return (
    <div className="chat-container">
      <div className="messages">
        {chat.map((msg, i) => (
          <div key={i}>
            <MessageBubble role={msg.role} text={msg.text} />
            {msg.sources && <SourceList sources={msg.sources} />}
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
