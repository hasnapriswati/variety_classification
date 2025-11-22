import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { supabase } from '../lib/supabaseClient'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000'

export default function LiveDetect() {
  const navigate = useNavigate()
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const snapCanvasRef = useRef(null)
  const streamRef = useRef(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [cameraOn, setCameraOn] = useState(false)

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }
    } catch (e) {
      setResult({ error: 'Tidak bisa mengakses kamera' })
      setCameraOn(false)
    }
  }

  const stopCamera = () => {
    try {
      streamRef.current?.getTracks()?.forEach(t => t.stop())
      if (videoRef.current) {
        videoRef.current.pause()
        videoRef.current.srcObject = null
      }
      const canvas = canvasRef.current
      if (canvas) {
        const ctx = canvas.getContext('2d')
        if (ctx) ctx.clearRect(0, 0, canvas.width || 0, canvas.height || 0)
      }
    } catch {}
  }

  useEffect(() => {
    let active = true
    if (cameraOn) startCamera()
    return () => { if (active) stopCamera() }
  }, [cameraOn])

  const drawDetections = (detections) => {
    const vid = videoRef.current
    const canvas = canvasRef.current
    if (!vid || !canvas) return
    const w = vid.videoWidth || 640
    const h = vid.videoHeight || 480
    canvas.width = w
    canvas.height = h
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, w, h)
    if (!detections || !Array.isArray(detections)) return
    ctx.strokeStyle = '#2F7E79'
    ctx.lineWidth = 2
    ctx.font = '14px Inter, sans-serif'
    ctx.fillStyle = 'rgba(47,126,121,0.85)'
    detections.forEach(d => {
      const [x1,y1,x2,y2] = d.bbox || []
      if (x2 > x1 && y2 > y1) {
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)
        const label = `${d.class_name || 'leaf'}${d.confidence != null ? ` ${(d.confidence*100).toFixed(0)}%` : ''}`
        const tw = ctx.measureText(label).width + 8
        ctx.fillRect(x1, Math.max(y1-22, 0), tw, 20)
        ctx.fillStyle = '#fff'
        ctx.fillText(label, x1 + 4, Math.max(y1-8, 12))
        ctx.fillStyle = 'rgba(47,126,121,0.85)'
      }
    })
  }

  const captureFrameBlob = async () => {
    const vid = videoRef.current
    const snap = snapCanvasRef.current
    if (!vid || !snap || vid.readyState < 2) return null
    const w = vid.videoWidth || 640
    const h = vid.videoHeight || 480
    snap.width = w
    snap.height = h
    const ctx = snap.getContext('2d')
    ctx.drawImage(vid, 0, 0, w, h)
    return new Promise((resolve) => snap.toBlob(b => resolve(b), 'image/jpeg', 0.9))
  }

  const snapAndNavigate = async () => {
    const blob = await captureFrameBlob()
    if (!blob) return
    const file = new File([blob], 'snapshot.jpg', { type: 'image/jpeg' })
    const url = URL.createObjectURL(blob)
    setPreviewUrl(url)
    try {
      const reader = new FileReader()
      reader.onload = () => { try { sessionStorage.setItem('last_preview', reader.result) } catch {} }
      reader.readAsDataURL(blob)
    } catch {}
    setLoading(true)
    setResult(null)
    const fd = new FormData()
    fd.append('file', file)
    try {
      const res = await fetch(`${API_URL}/predict`, { method: 'POST', body: fd })
      const data = await res.json()
      setLoading(false)
      if (data?.success) {
        try { drawDetections(data?.yolo?.detections || []) } catch {}
        const out = !!data.is_out_of_scope || String(data.variety || '').trim().toLowerCase() === 'unknown'
        try {
          const raw = JSON.stringify(data)
          sessionStorage.setItem('last_result', raw)
          localStorage.setItem('last_result', raw)
          if (!out) {
            const { data: userResponse } = await supabase.auth.getUser()
            const user = userResponse?.user
            if (user) {
              await supabase.from('predictions').insert([
                { user_id: user.id, variety: data.variety, confidence: data.confidence_percentage, image_url: '' }
              ])
            }
          }
        } catch {}
        if (out) {
          const noLeaf = Array.isArray(data?.yolo?.detections) ? data.yolo.detections.length === 0 : false
          setResult({
            error: noLeaf ? 'Daun tidak terdeteksi pada gambar.' : 'Gambar di luar cakupan daun cabai (13 varietas) yang diteliti.',
            reasons: Array.isArray(data?.out_of_scope_reasons) ? data.out_of_scope_reasons : [],
            noLeaf
          })
        } else {
          try {
            if (data.preview_base64) {
              sessionStorage.setItem('last_preview', data.preview_base64)
              localStorage.setItem('last_preview', data.preview_base64)
            } else {
              const reader = new FileReader()
              reader.onload = () => { try { sessionStorage.setItem('last_preview', reader.result); localStorage.setItem('last_preview', reader.result) } catch {} }
              reader.readAsDataURL(blob)
            }
          } catch {}
          navigate('/results', { state: { result: data, previewUrl: url } })
        }
      } else {
        setResult({ error: data?.error || 'Prediksi gagal' })
      }
    } catch (e) {
      setLoading(false)
      setResult({ error: e.message })
    }
  }

  return (
    <div className="container">
      <div className="card">
        <h1 className="card-title">Deteksi Kamera</h1>
        <p className="muted">Arahkan kamera ke daun, tekan Deteksi untuk melihat kotak.</p>
        <div className="video-wrap">
          {cameraOn ? (
            <>
              <video ref={videoRef} className="video-frame" muted playsInline />
              <canvas ref={canvasRef} className="video-overlay" />
              <canvas ref={snapCanvasRef} style={{ display: 'none' }} />
                
              {loading && <div className="loader-veil"><span className="badge">Memproses…</span></div>}
            </>
          ) : (
            <div className="placeholder">
              <div className="placeholder-inner">
                <div className="ph-title">Siap Deteksi di Lapangan</div>
                <div className="ph-sub">Aktifkan kamera untuk mulai</div>
                <div className="ph-actions">
                  <button className="btn" onClick={() => setCameraOn(true)}>Mulai Kamera</button>
                </div>
              </div>
            </div>
          )}
        </div>
        {cameraOn && (
          <div className="actions-row">
            <button className="btn" onClick={snapAndNavigate} disabled={loading}>{loading ? 'Memproses...' : 'Jepret & Prediksi'}</button>
            <a className="link link-danger" onClick={() => setCameraOn(false)}>Matikan Kamera</a>
          </div>
        )}
        {result?.error && (
          <div className="section" style={{ marginTop: 12 }}>
            <div className="section-title">Klasifikasi tidak dapat dilakukan</div>
            <p className="error" role="alert">{result.error}</p>
            {result.noLeaf && (
              <div className="muted small">Pastikan daun cabai terlihat jelas, tidak blur, dan berada di tengah bingkai.</div>
            )}
            
            <div className="actions" style={{ marginTop: 10 }}>
              <button className="btn" onClick={() => setResult(null)}>Ambil Ulang</button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}