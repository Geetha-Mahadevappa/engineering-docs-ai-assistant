export default function MessageBubble({ role, text }) {
  return (
    <div className={`bubble ${role}`}>
      <strong>{role === "user" ? "You" : "Assistant"}</strong>
      <p>{text}</p>
    </div>
  );
}
