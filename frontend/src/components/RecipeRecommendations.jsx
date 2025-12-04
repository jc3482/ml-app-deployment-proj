import React, { useState } from 'react'
import './RecipeRecommendations.css'

function RecipeRecommendations({ recommendations, loading }) {
  if (loading) {
    return (
      <div className="card">
        <div className="loading">Loading recommendations...</div>
      </div>
    )
  }

  if (recommendations.length === 0) {
    return (
      <div className="card">
        <h2>Recipe Recommendations</h2>
        <p className="empty-message">Upload an image and click 'Get Recommendations' to see recipe suggestions.</p>
      </div>
    )
  }

  return (
    <div className="card recommendations-card">
      <h2>Recipe Recommendations</h2>
      <div className="recommendations-list">
        {recommendations.map((recipe, index) => (
          <div key={index} className="recipe-card">
            <div className="recipe-header">
              <h3>
                {index + 1}. {recipe.title}
                <span className="score-badge">{recipe.score.toFixed(1)}%</span>
              </h3>
            </div>
            <div className="recipe-ingredients">
              <strong>Ingredients:</strong>{' '}
              <span>{recipe.normalized_ingredients.join(', ')}</span>
            </div>
            <div className="recipe-instructions">
              <strong>Instructions:</strong>
              {recipe.instructions.length > 300 ? (
                <details>
                  <summary>{recipe.instructions.substring(0, 300)}...</summary>
                  <div className="instructions-full">
                    {recipe.instructions}
                  </div>
                </details>
              ) : (
                <div className="instructions-full">
                  {recipe.instructions}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default RecipeRecommendations

