import { supabase } from './supabase'

const BASE = process.env.NEXT_PUBLIC_API_URL

/**
 * Authenticated fetch wrapper.
 * Injects the Supabase session JWT into every request.
 * Throws an Error with the response body text if the request fails.
 */
export async function apiFetch(path: string, init: RequestInit = {}) {
  const {
    data: { session },
  } = await supabase.auth.getSession()

  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...init.headers,
      // Authorization is appended last so it cannot be overridden by callers
      ...(session?.access_token
        ? { Authorization: `Bearer ${session.access_token}` }
        : {}),
    },
  })

  if (!res.ok) {
    const text = await res.text()
    throw new Error(text || `HTTP ${res.status}`)
  }

  return res.json()
}

/**
 * Upload a PDF file to a specific project via R2-backed endpoint.
 * Returns the drawing record created in Supabase.
 */
export async function uploadDrawing(projectId: string, file: File) {
  const {
    data: { session },
  } = await supabase.auth.getSession()

  const formData = new FormData()
  formData.append('file', file)

  const res = await fetch(`${BASE}/projects/${projectId}/upload`, {
    method: 'POST',
    headers: {
      ...(session?.access_token
        ? { Authorization: `Bearer ${session.access_token}` }
        : {}),
    },
    body: formData,
  })

  if (!res.ok) {
    const text = await res.text()
    throw new Error(text || `HTTP ${res.status}`)
  }

  return res.json()
}
