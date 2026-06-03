import React, { useState, useRef } from 'react';
import axios from 'axios';
import './App.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [caption, setCaption] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const handleFile = (file) => {
    if (!file) return;
    const allowed = ['image/jpeg','image/jpg','image/png','image/webp','image/bmp'];
    if (!allowed.includes(file.type)) {
      setError('Please upload JPG, PNG, WEBP, or BMP');
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      setError('File must be under 10MB');
      return;
    }
    setError('');
    setSelectedFile(file);
    setCaption('');
    const reader = new FileReader();
    reader.onloadend = () => setPreviewUrl(reader.result);
    reader.readAsDataURL(file);
  };

  const handleDrag = (e) => {
    e.preventDefault(); e.stopPropagation();
    setDragActive(e.type === 'dragenter' || e.type === 'dragover');
  };

  const handleDrop = (e) => {
    e.preventDefault(); e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files?.[0]) handleFile(e.dataTransfer.files[0]);
  };

  const handleChange = (e) => {
    if (e.target.files?.[0]) handleFile(e.target.files[0]);
  };

  const handleSubmit = async () => {
    if (!selectedFile) { setError('Select an image first'); return; }
    setLoading(true); setError(''); setCaption('');
    const formData = new FormData();
    formData.append('file', selectedFile);
    try {
      const res = await axios.post(`${API_URL}/predict`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 60000,
      });
      if (res.data.success) {
        setCaption(res.data.caption);
      }
    } catch (err) {
      if (err.response?.data?.detail) setError(err.response.data.detail);
      else if (err.code === 'ERR_NETWORK') setError('Backend not running on port 8000');
      else setError('Something went wrong');
    } finally { setLoading(false); }
  };

  const clearAll = () => {
    setSelectedFile(null); setPreviewUrl(null); setCaption(''); setError('');
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <h1>Image Caption Generator</h1>
          <p>Upload an image and let AI describe it</p>
        </div>
      </header>
      <main className="main">
        <div className="container">
          <div className={`upload-card ${dragActive ? 'drag-active' : ''} ${previewUrl ? 'has-preview' : ''}`}
            onDragEnter={handleDrag} onDragLeave={handleDrag} onDragOver={handleDrag} onDrop={handleDrop}>
            {!previewUrl ? (
              <div className="upload-placeholder">
                <div className="upload-icon">📤</div>
                <p className="upload-text"><strong>Drag & drop</strong> an image, or <span className="browse-link" onClick={() => fileInputRef.current?.click()}>browse</span></p>
                <p className="upload-hint">JPG, PNG, WEBP, BMP (max 10MB)</p>
              </div>
            ) : (
              <div className="preview-container">
                <img src={previewUrl} alt="Preview" className="preview-image" />
                <button className="remove-btn" onClick={clearAll}>✕</button>
              </div>
            )}
            <input ref={fileInputRef} type="file" accept="image/*" onChange={handleChange} className="file-input" />
          </div>

          {error && <div className="error-message"><span>⚠️</span> {error}</div>}

          {previewUrl && (
            <button className={`generate-btn ${loading ? 'loading' : ''}`} onClick={handleSubmit} disabled={loading}>
              {loading ? <><span className="spinner"></span> Generating...</> : <>✨ Generate Caption</>}
            </button>
          )}

          {caption && (
            <div className="result-card">
              <div className="result-header"><span className="result-badge">AI Generated</span></div>
              <blockquote className="caption-text">“{caption}”</blockquote>
            </div>
          )}
        </div>
      </main>
      <footer className="footer"><p>Powered by Trained Vision-Language Model | React + FastAPI</p></footer>
    </div>
  );
}

export default App;
