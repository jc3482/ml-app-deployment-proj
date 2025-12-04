import React, { useState } from 'react'
import { detectIngredients, getRecommendations } from '../services/api'
import './ImageUpload.css'

function ImageUpload({ onDetect, onRecommend, detectedIngredients, pantry, topK, dietaryFilter, loading, setLoading }) {
  const [image, setImage] = useState(null)
  const [imagePreview, setImagePreview] = useState(null)
  const [error, setError] = useState(null)

  const handleFileChange = (e) => {
    const file = e.target.files[0]
    if (file) {
      setImage(file)
      setError(null)
      const reader = new FileReader()
      reader.onloadend = () => {
        setImagePreview(reader.result)
      }
      reader.readAsDataURL(file)
    }
  }

  const handleDetect = async () => {
    if (!image) {
      setError('Please upload an image first')
      return
    }

    setLoading(true)
    setError(null)
    try {
      const result = await detectIngredients(image)
      onDetect(result.canonical_ingredients || [])
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleRecommend = async () => {
    if (!image && detectedIngredients.length === 0 && pantry.length === 0) {
      setError('Please upload an image or add ingredients to pantry')
      return
    }

    setLoading(true)
    setError(null)
    try {
      const result = await getRecommendations(
        image,
        detectedIngredients,
        pantry,
        topK,
        dietaryFilter
      )
      onRecommend(result.recommendations || [])
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="card image-upload">
      <h2>Upload Image</h2>
      <div className="upload-area">
        {imagePreview ? (
          <img src={imagePreview} alt="Preview" className="preview-image" />
        ) : (
          <div className="upload-placeholder">
            <p>Click to upload or drag and drop</p>
            <p className="upload-hint">PNG, JPG, GIF up to 10MB</p>
          </div>
        )}
        <input
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="file-input"
        />
      </div>
      {error && <div className="error">{error}</div>}
      <div className="button-group">
        <button 
          className="btn btn-primary" 
          onClick={handleDetect}
          disabled={loading || !image}
        >
          {loading ? 'Detecting...' : 'Detect Ingredients'}
        </button>
        <button 
          className="btn btn-primary" 
          onClick={handleRecommend}
          disabled={loading || (!image && detectedIngredients.length === 0 && pantry.length === 0)}
        >
          {loading ? 'Loading...' : 'Get Recommendations'}
        </button>
      </div>
    </div>
  )
}

export default ImageUpload

