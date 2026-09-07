import { useState, useEffect } from 'react'
import './App.css'

function App() {
  const [healthStatus, setHealthStatus] = useState("Checking backend health...")

  useEffect(() => {
    fetch('http://127.0.0.1:8000/health')
      .then(res => res.json())
      .then(data => setHealthStatus(`Backend says: ${data.message}`))
      .catch(err => setHealthStatus(`Backend error: ${err.message}`))
  }, [])

  return (
    <div className="App">
      <h1>Semantic Search App Shell</h1>
      <div className="card">
        <p>
          Status: {healthStatus}
        </p>
      </div>
    </div>
  )
}

export default App
