import { useState } from "react";

export default function SearchBar() {
  const [query, setQuery] = useState("");

  async function search() {
    const res = await fetch("http://localhost:8000/search?q=" + query);
    const json = await res.json();
    console.log(json);
  }

  return (
    <div className="search-bar">
      <input
        className="search-input"
        value={query}
        onChange={e => setQuery(e.target.value)}
        placeholder="Search your documents..."
      />
      <button className="search-btn" onClick={search}>
        Search
      </button>
    </div>
  );
}
