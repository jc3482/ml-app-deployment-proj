import React, { useState } from 'react'
import { addToPantry, removeFromPantry, clearPantry } from '../services/api'
import './PantryList.css'

function PantryList({ pantry, setPantry }) {
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)

  const handleAdd = async () => {
    if (!input.trim()) return

    const ingredients = input.split(',').map(i => i.trim()).filter(i => i)
    if (ingredients.length === 0) return

    setLoading(true)
    try {
      const updated = await addToPantry(ingredients)
      setPantry(updated)
      setInput('')
    } catch (err) {
      alert(`Error: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }

  const handleRemove = async (ingredient) => {
    setLoading(true)
    try {
      const updated = await removeFromPantry(ingredient)
      setPantry(updated)
    } catch (err) {
      alert(`Error: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }

  const handleClear = async () => {
    if (!confirm('Clear all pantry items?')) return

    setLoading(true)
    try {
      const updated = await clearPantry()
      setPantry(updated)
    } catch (err) {
      alert(`Error: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="card">
      <h2>My Pantry List</h2>
      <div className="pantry-input-group">
        <input
          type="text"
          className="input"
          placeholder="Enter ingredients (e.g., milk, eggs, flour)"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleAdd()}
          disabled={loading}
        />
        <div className="button-group">
          <button
            className="btn btn-primary"
            onClick={handleAdd}
            disabled={loading || !input.trim()}
          >
            Add to Pantry
          </button>
          <button
            className="btn btn-secondary"
            onClick={handleClear}
            disabled={loading || pantry.length === 0}
          >
            Clear Pantry
          </button>
        </div>
      </div>
      <details className="pantry-details">
        <summary>
          <h3 style={{ display: 'inline', margin: 0 }}>
            My Pantry ({pantry.length} items)
          </h3>
        </summary>
        {pantry.length === 0 ? (
          <p className="empty-message">Your pantry is empty. Add ingredients above.</p>
        ) : (
          <ul className="pantry-list">
            {pantry.map((ingredient, index) => (
              <li key={index} className="pantry-item">
                <span>{ingredient}</span>
                <button
                  className="remove-btn"
                  onClick={() => handleRemove(ingredient)}
                  title={`Remove ${ingredient}`}
                  disabled={loading}
                >
                  ×
                </button>
              </li>
            ))}
          </ul>
        )}
      </details>
    </div>
  )
}

export default PantryList

