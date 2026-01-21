export default function UploadWidget() {
  async function handleUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    const form = new FormData();
    form.append("file", file);

    await fetch("http://localhost:8000/upload", {
      method: "POST",
      body: form
    });

    alert("Uploaded!");
  }

  return (
    <div className="upload-box">
      <p>Upload engineering documents</p>
      <input type="file" onChange={handleUpload} />
    </div>
  );
}
