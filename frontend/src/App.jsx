import React, { useState, useEffect } from 'react'
import Header from './components/Header'
import ImageUpload from './components/ImageUpload'
import DetectedIngredients from './components/DetectedIngredients'
import PantryList from './components/PantryList'
import RecipeRecommendations from './components/RecipeRecommendations'
import History from './components/History'
import Settings from './components/Settings'
import { loadPantry } from './services/api'
import './App.css'

function App() {
  const [detectedIngredients, setDetectedIngredients] = useState([])
  const [pantry, setPantry] = useState([])
  const [recommendations, setRecommendations] = useState([])
  const [history, setHistory] = useState([])
  const [topK, setTopK] = useState(10)
  const [dietaryFilter, setDietaryFilter] = useState('None')
  const [loading, setLoading] = useState(false)

  // Load pantry on mount
  useEffect(() => {
    loadPantry()
      .then(setPantry)
      .catch((err) => {
        console.error('Failed to load pantry:', err)
        // Don't crash the app if pantry fails to load
        setPantry([])
      })
  }, [])

  return (
    <div className="app">
      <Header />
      <div className="container">
        <div className="main-content">
          <div className="left-column">
            <ImageUpload
              onDetect={setDetectedIngredients}
              onRecommend={setRecommendations}
              detectedIngredients={detectedIngredients}
              pantry={pantry}
              topK={topK}
              dietaryFilter={dietaryFilter}
              loading={loading}
              setLoading={setLoading}
            />
            <Settings
              topK={topK}
              setTopK={setTopK}
              dietaryFilter={dietaryFilter}
              setDietaryFilter={setDietaryFilter}
            />
          </div>
          <div className="right-column">
            <DetectedIngredients
              ingredients={detectedIngredients}
              onRemove={(ingredient) => {
                setDetectedIngredients(prev => prev.filter(i => i !== ingredient))
              }}
            />
            <PantryList
              pantry={pantry}
              setPantry={setPantry}
            />
          </div>
        </div>
        <RecipeRecommendations
          recommendations={recommendations}
          loading={loading}
        />
        <History
          history={history}
          setHistory={setHistory}
        />
      </div>
    </div>
  )
}

export default App

