import React from 'react'
import './Settings.css'

function Settings({ topK, setTopK, dietaryFilter, setDietaryFilter }) {
  return (
    <div className="card">
      <h2>Settings</h2>
      <div className="setting-group">
        <label>
          Number of Recipes
          <input
            type="range"
            min="1"
            max="20"
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
            className="slider"
          />
          <span className="slider-value">{topK}</span>
        </label>
      </div>
      <div className="setting-group">
        <label>
          Dietary Filter
          <select
            value={dietaryFilter}
            onChange={(e) => setDietaryFilter(e.target.value)}
            className="select"
          >
            <option value="None">None</option>
            <option value="vegan">Vegan</option>
            <option value="vegetarian">Vegetarian</option>
            <option value="dairy-free">Dairy-Free</option>
            <option value="gluten-free">Gluten-Free</option>
          </select>
        </label>
      </div>
    </div>
  )
}

export default Settings

