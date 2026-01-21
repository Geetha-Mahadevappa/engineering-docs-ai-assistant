import { useEffect, useState } from "react";

export default function Sidebar() {
  const [sessions, setSessions] = useState([]);

  useEffect(() => {
    fetch("http://localhost:8000/sessions")
      .then(res => res.json())
      .then(setSessions);
  }, []);

  return (
    <div className="sidebar">
      <h3>Conversations</h3>
      {sessions.map(s => (
        <div key={s.id} className="session-item">
          {s.title || "Untitled session"}
        </div>
      ))}
    </div>
  );
}
