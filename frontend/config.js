/**
 * ExoDip Backend URL config.
 * For local dev: leave empty (uses relative paths).
 * For Vercel: set window.EXODIP_BACKEND_URL = 'https://your-app.onrender.com'
 * via a Vercel Edge Config or inject it with a tiny inline script in index.html.
 */
const BACKEND_URL = (typeof window !== 'undefined' && window.EXODIP_BACKEND_URL) || '';
