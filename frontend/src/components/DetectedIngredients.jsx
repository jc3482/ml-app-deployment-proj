import React from 'react'
import './DetectedIngredients.css'

function DetectedIngredients({ ingredients, onRemove }) {
  if (ingredients.length === 0) {
    return (
      <div className="card">
        <h2>Detected Ingredients</h2>
        <p className="empty-message">Upload an image and click 'Detect Ingredients' to see detected items.</p>
      </div>
    )
  }

  return (
    <div className="card">
      <details className="ingredients-details">
        <summary>
          <h2 style={{ display: 'inline', margin: 0 }}>
            Detected Ingredients ({ingredients.length} items)
          </h2>
        </summary>
        <ul className="ingredients-list">
          {ingredients.map((ingredient, index) => (
            <li key={index} className="ingredient-item">
              <span>{ingredient}</span>
              <button
                className="remove-btn"
                onClick={() => onRemove(ingredient)}
                title={`Remove ${ingredient}`}
              >
                ×
              </button>
            </li>
          ))}
        </ul>
      </details>
    </div>
  )
}

export default DetectedIngredients

