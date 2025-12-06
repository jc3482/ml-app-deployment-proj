# SmartPantry Frontend (React)

Modern React frontend for SmartPantry Recipe Recommender.

## Features

- ✅ Image upload and ingredient detection
- ✅ Pantry management (add/remove ingredients)
- ✅ Recipe recommendations with dietary filters
- ✅ History tracking
- ✅ Beautiful beige/tan UI matching the design
- ✅ Responsive layout
- ✅ Delete buttons (×) for ingredients

## Setup

### Install Dependencies

```bash
cd frontend
npm install
```

### Development

```bash
npm run dev
```

Frontend will run on `http://localhost:3000`

### Build for Production

```bash
npm run build
```

Output will be in `dist/` directory.

## API Configuration

The frontend expects the API to be running on `http://localhost:8000`.

The Vite config includes a proxy, so API calls to `/api/*` will be forwarded to `http://localhost:8000/*`.

## Project Structure

```
frontend/
├── src/
│   ├── components/       # React components
│   │   ├── Header.jsx
│   │   ├── ImageUpload.jsx
│   │   ├── DetectedIngredients.jsx
│   │   ├── PantryList.jsx
│   │   ├── RecipeRecommendations.jsx
│   │   ├── History.jsx
│   │   └── Settings.jsx
│   ├── services/
│   │   └── api.js        # API client
│   ├── App.jsx           # Main app component
│   ├── App.css
│   ├── main.jsx          # Entry point
│   └── index.css         # Global styles
├── index.html
├── package.json
└── vite.config.js
```

## Running with Backend

### Terminal 1: Start FastAPI Backend

```bash
cd /path/to/ml-app-deployment-proj
uvicorn app.api_extended:app --reload --port 8000
```

### Terminal 2: Start React Frontend

```bash
cd frontend
npm run dev
```

Then open `http://localhost:3000` in your browser.

## Deployment

### Option 1: Static Hosting (Recommended)

1. Build the frontend:
   ```bash
   npm run build
   ```

2. Deploy `dist/` folder to:
   - Netlify
   - Vercel
   - GitHub Pages
   - AWS S3 + CloudFront
   - Any static hosting service

3. Configure API endpoint:
   - Update `API_BASE` in `src/services/api.js` to point to your production API URL
   - Or use environment variables

### Option 2: Serve with Backend

You can serve the built frontend from FastAPI:

```python
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI

app = FastAPI()

# Serve static files
app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="static")
```

## Environment Variables

Create `.env` file for different environments:

```env
VITE_API_BASE=http://localhost:8000
```

Then update `src/services/api.js`:

```javascript
const API_BASE = import.meta.env.VITE_API_BASE || '/api'
```

