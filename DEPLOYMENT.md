# 🚀 Deployment Guide - NewsSense

This guide walks you through deploying the NewsSense application:
- **Frontend**: Vercel (React + Vite)
- **Backend**: Render (FastAPI + ML Models)

---

## 📋 Prerequisites

- [GitHub Account](https://github.com)
- [Vercel Account](https://vercel.com) (sign up with GitHub)
- [Render Account](https://render.com) (sign up with GitHub)
- Repository pushed to GitHub: `https://github.com/Naveen-Arul/NewsSense.git`

---

## 🔧 Part 1: Deploy Backend to Render

### Step 1: Prepare Backend for Deployment

Before deploying, ensure you have trained models in the `models/` folder:

```bash
# Run this locally if you haven't already
python news_classification.py
```

This creates:
- `naive_bayes_model.pkl`
- `logistic_regression_model.pkl`
- `svm_model.pkl`
- `random_forest_model.pkl`
- `tfidf_vectorizer.pkl`

### Step 2: Push Models to GitHub

The models folder needs to be committed to your repository:

```bash
git add models/
git commit -m "Add trained ML models for deployment"
git push
```

### Step 3: Create Render Web Service

1. **Go to Render Dashboard**: https://dashboard.render.com/
2. **Click "New +"** → Select **"Web Service"**
3. **Connect GitHub Repository**:
   - Click "Connect account" if first time
   - Select repository: `Naveen-Arul/NewsSense`
   - Click "Connect"

4. **Configure Web Service**:
   ```
   Name: newssense-api
   Region: Choose closest to you (e.g., Singapore, Oregon)
   Branch: master
   Root Directory: (leave empty)
   Runtime: Python 3
   Build Command: pip install -r requirements-api.txt && python news_classification.py
   Start Command: uvicorn app:app --host 0.0.0.0 --port $PORT
   ```

5. **Select Plan**:
   - Choose **"Free"** plan (for testing)
   - Note: Free tier sleeps after 15 min inactivity

6. **Advanced Settings** (optional):
   - Add environment variable if needed:
     ```
     PORT = 8000
     PYTHON_VERSION = 3.11.0
     ```

7. **Click "Create Web Service"**

### Step 4: Wait for Deployment

- Render will automatically:
  1. Install dependencies from `requirements-api.txt`
  2. Run `news_classification.py` to train models
  3. Start the FastAPI server with Uvicorn
  
- **Initial deployment takes 5-10 minutes** (training models)
- Monitor logs in real-time on Render dashboard

### Step 5: Get Backend URL

Once deployed, you'll get a URL like:
```
https://newssense-api.onrender.com
```

**Test the API**:
- Visit: `https://newssense-api.onrender.com/models`
- Should return JSON with model metrics

---

## 🌐 Part 2: Deploy Frontend to Vercel

### Step 1: Update Frontend API URL

Create a `.env.production` file in the `frontend/` folder:

```bash
cd frontend
echo VITE_API_URL=https://newssense-api.onrender.com > .env.production
```

Replace `https://newssense-api.onrender.com` with your **actual Render URL** from Part 1.

### Step 2: Commit Environment Config

```bash
git add frontend/.env.production
git commit -m "Add production API URL for Vercel deployment"
git push
```

### Step 3: Deploy to Vercel

1. **Go to Vercel Dashboard**: https://vercel.com/dashboard
2. **Click "Add New..."** → Select **"Project"**
3. **Import Git Repository**:
   - Select `Naveen-Arul/NewsSense`
   - Click "Import"

4. **Configure Project**:
   ```
   Framework Preset: Vite
   Root Directory: frontend
   Build Command: npm run build (auto-detected)
   Output Directory: dist (auto-detected)
   Install Command: npm install (auto-detected)
   ```

5. **Add Environment Variable**:
   - Click "Environment Variables"
   - Add:
     ```
     Name: VITE_API_URL
     Value: https://newssense-api.onrender.com
     ```
   - Replace with your Render URL

6. **Click "Deploy"**

### Step 4: Wait for Deployment

- Vercel builds and deploys in **1-2 minutes**
- You'll get a URL like:
  ```
  https://newssense-xyz123.vercel.app
  ```

### Step 5: Test Your Deployed App

1. Visit your Vercel URL
2. Click "Try Classification"
3. Enter news text and classify
4. Verify predictions are working

---

## ⚙️ Part 3: Update Backend CORS (Important!)

Your backend needs to allow requests from your Vercel domain.

### Update app.py

Open `app.py` and update the CORS middleware:

```python
# Current CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Change to**:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",  # Local development
        "https://newssense-xyz123.vercel.app",  # Your Vercel URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Push changes**:

```bash
git add app.py
git commit -m "Update CORS to allow Vercel frontend"
git push
```

Render will **auto-redeploy** when you push to GitHub.

---

## 🎉 Deployment Complete!

Your app is now live:

- **Frontend**: `https://newssense-xyz123.vercel.app`
- **Backend**: `https://newssense-api.onrender.com`

---

## 🔍 Troubleshooting

### Backend Issues

**Problem**: Build fails with "ModuleNotFoundError"
- **Solution**: Check `requirements-api.txt` has all dependencies with versions

**Problem**: Models not loading (404 error)
- **Solution**: Ensure `models/` folder is committed to GitHub
- Run `git add models/ -f` if `.gitignore` blocks it

**Problem**: API returns 500 errors
- **Solution**: Check Render logs for Python errors
- Verify NLTK data downloads correctly

### Frontend Issues

**Problem**: API calls fail (CORS error)
- **Solution**: Update CORS origins in `app.py` to include Vercel URL

**Problem**: Environment variable not working
- **Solution**: Ensure `VITE_API_URL` is set in Vercel dashboard
- Redeploy after adding env vars

**Problem**: Build fails
- **Solution**: Check `package.json` dependencies
- Ensure Node.js version compatibility

---

## 💰 Cost Breakdown

| Service | Free Tier | Paid Plan |
|---------|-----------|-----------|
| **Vercel** | ✅ Unlimited bandwidth, 100 GB/month | $20/month Pro |
| **Render** | ✅ 750 hours/month, sleeps after 15 min | $7/month for always-on |

**Total for Free Tier**: $0/month  
**Total for Paid**: ~$27/month

---

## 🔄 Update Workflow

### Update Frontend

1. Make changes locally in `frontend/`
2. Test with `npm run dev`
3. Push to GitHub:
   ```bash
   git add .
   git commit -m "Update frontend"
   git push
   ```
4. Vercel auto-deploys in ~1 minute

### Update Backend

1. Make changes locally in `app.py` or `news_classification.py`
2. Test with `python app.py`
3. Push to GitHub:
   ```bash
   git add .
   git commit -m "Update backend"
   git push
   ```
4. Render auto-deploys in ~3-5 minutes

---

## 📊 Monitoring

### Vercel
- **Logs**: https://vercel.com/dashboard → Your Project → Deployments
- **Analytics**: Built-in traffic analytics

### Render
- **Logs**: https://dashboard.render.com → Your Service → Logs
- **Metrics**: CPU, Memory, Request stats

---

## 🌟 Optional Enhancements

### Custom Domain (Vercel)
1. Go to Vercel Dashboard → Settings → Domains
2. Add your domain (e.g., `newssense.com`)
3. Update DNS records as instructed

### Environment-Specific Settings
- Create `.env.development` for local dev
- Create `.env.production` for production
- Keep API keys secure in Vercel/Render dashboards

---

## 📝 Quick Reference

### Local Development
```bash
# Backend
python app.py

# Frontend
cd frontend
npm run dev
```

### Production URLs
- **Frontend**: https://your-vercel-url.vercel.app
- **Backend API**: https://your-render-url.onrender.com

---

**Need Help?** Check:
- [Vercel Docs](https://vercel.com/docs)
- [Render Docs](https://render.com/docs)
- GitHub Issues: https://github.com/Naveen-Arul/NewsSense/issues
