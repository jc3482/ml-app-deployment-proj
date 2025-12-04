import React, { useEffect } from 'react'
import { loadHistory, clearHistory } from '../services/api'
import './History.css'

function History({ history, setHistory }) {
  useEffect(() => {
    loadHistory().then(setHistory).catch(console.error)
  }, [setHistory])

  const handleClear = async () => {
    if (!confirm('Clear all history?')) return
    try {
      const cleared = await clearHistory()
      setHistory(cleared)
    } catch (err) {
      alert(`Error: ${err.message}`)
    }
  }

  if (history.length === 0) {
    return (
      <div className="card">
        <details className="history-details">
          <summary>
            <h2 style={{ display: 'inline', margin: 0 }}>History (0 items)</h2>
          </summary>
          <p className="empty-message">No history yet. Start detecting ingredients to see your history here.</p>
        </details>
      </div>
    )
  }

  return (
    <div className="card">
      <div className="history-header">
        <details className="history-details">
          <summary>
            <h2 style={{ display: 'inline', margin: 0 }}>History ({history.length} items)</h2>
          </summary>
          <div className="history-list">
            {history.slice(0, 10).map((record, index) => {
              const date = new Date(record.timestamp)
              const timeStr = date.toLocaleString()
              return (
                <div key={index} className="history-item">
                  <p className="history-time">{timeStr}</p>
                  <p className="history-ingredients">
                    <strong>Ingredients:</strong> {record.ingredients.slice(0, 10).join(', ')}
                    {record.ingredients.length > 10 && '...'}
                  </p>
                  {record.recipes && record.recipes.length > 0 && (
                    <p className="history-top-recipe">
                      Top: {record.recipes[0].title}
                    </p>
                  )}
                </div>
              )
            })}
          </div>
        </details>
        <button className="btn btn-secondary" onClick={handleClear}>
          Clear History
        </button>
      </div>
    </div>
  )
}

export default History

