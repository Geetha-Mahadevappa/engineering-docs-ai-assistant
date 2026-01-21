import Sidebar from "../components/Sidebar.jsx";
import UploadWidget from "../components/UploadWidget.jsx";
import SearchBar from "../components/SearchBar.jsx";
import Chat from "../components/Chat.jsx";

export default function App() {
  return (
    <div className="app-layout">
      <Sidebar />

      <div className="main-panel">
        <UploadWidget />
        <SearchBar />
        <Chat />
      </div>
    </div>
  );
}
