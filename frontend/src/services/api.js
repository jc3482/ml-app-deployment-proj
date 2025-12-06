const API_BASE = '/api'

async function apiCall(endpoint, options = {}) {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
    throw new Error(error.detail || `HTTP ${response.status}`)
  }

  return response.json()
}

// Image detection
export async function detectIngredients(file) {
  const formData = new FormData()
  formData.append('file', file)
  
  const response = await fetch(`${API_BASE}/detect`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
    throw new Error(error.detail || `HTTP ${response.status}`)
  }

  return response.json()
}

// Recipe recommendations
export async function getRecommendations(file, detectedIngredients, pantry, topK, dietaryFilter) {
  const body = {
    detected_ingredients: detectedIngredients || [],
    pantry_ingredients: pantry || [],
    top_k: topK,
    dietary_filter: dietaryFilter || 'None',
  }

  // If we have a file, use multipart/form-data
  if (file) {
    const formData = new FormData()
    formData.append('file', file)
    // Send JSON as a string in form data (field name must match backend)
    formData.append('request', JSON.stringify(body))
    
    const response = await fetch(`${API_BASE}/recommend`, {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
      throw new Error(error.detail || `HTTP ${response.status}`)
    }

    return response.json()
  } else {
    // No file, send as JSON
    const response = await fetch(`${API_BASE}/recommend`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    })

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
      throw new Error(error.detail || `HTTP ${response.status}`)
    }

    return response.json()
  }
}

// Pantry management
export async function loadPantry() {
  try {
    const response = await apiCall('/pantry/list')
    return response.pantry || []
  } catch (error) {
    console.error('Error loading pantry:', error)
    // Return empty array if API fails
    return []
  }
}

export async function addToPantry(ingredients) {
  const response = await apiCall('/pantry/add', {
    method: 'POST',
    body: JSON.stringify({ ingredients }),
  })
  return response.pantry || []
}

export async function removeFromPantry(ingredient) {
  const response = await apiCall(`/pantry/remove/${encodeURIComponent(ingredient)}`, {
    method: 'DELETE',
  })
  return response.pantry || []
}

export async function clearPantry() {
  const response = await apiCall('/pantry/clear', {
    method: 'POST',
  })
  return response.pantry || []
}

// History
export async function loadHistory() {
  const response = await apiCall('/history')
  return response.history || []
}

export async function clearHistory() {
  await apiCall('/history/clear', {
    method: 'POST',
  })
  return []
}

