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
  
  const [cropRect, setCropRect] = useState(null)
  const [cropStart, setCropStart] = useState(null)
  const [lastDetections, setLastDetections] = useState([])

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
    try {
      const fd = new FormData()
      fd.append('file', file)
      const res = await fetch(`${API_URL}/predict`, { method: 'POST', body: fd })
      const data = await res.json()
      setLoading(false)
      if (data?.success) {
        try { drawDetections(data?.yolo?.detections || []) } catch {}
        const dets = Array.isArray(data?.yolo?.detections) ? data.yolo.detections : []
        setLastDetections(dets)
        const out = (!!data.is_out_of_scope)
          || (String(data.variety || '').trim().toLowerCase() === 'unknown')
          || (dets.length === 0)
        try {
          const raw = JSON.stringify(data)
          sessionStorage.setItem('last_result', raw)
          localStorage.setItem('last_result', raw)
          const { data: sessionData } = await supabase.auth.getSession()
          const user = sessionData?.session?.user
          if (user && !out) {
            const confNum = (typeof data.confidence === 'number' && !isNaN(data.confidence))
              ? data.confidence
              : (typeof data.confidence_percentage === 'string'
                  ? (parseFloat((data.confidence_percentage.match(/([0-9]+(?:\.[0-9]+)?)/)?.[1] || '0')) / 100)
                  : 0)
            let payload = {
              user_id: user.id,
              filename: 'snapshot.jpg',
              confidence: confNum,
              confidence_percentage: data.confidence_percentage || null,
              predicted_class: data.variety,
              api_version: data.api_version || null,
              model_version: JSON.stringify({ pipeline: data.decision_rule || 'efficientnet_only' }),
              morphology_info: data.morphology_info || null,
              measurement_quality: data.measurement_quality || null,
              variety_characteristics: data.variety_characteristics || null
            }
            let error = null
            for (let i = 0; i < 6; i++) {
              const r = await supabase.from('predictions').insert([payload])
              error = r.error
              if (!error) break
              const msg = String(error.message || '')
              const m = msg.match(/column \"([^\"]+)\"/i)
              if (m && m[1]) {
                delete payload[m[1]]
                continue
              }
              if (payload.predicted_class && !payload.variety) {
                payload = { ...payload, variety: payload.predicted_class }
              }
              payload = {
                user_id: user.id,
                filename: 'snapshot.jpg',
                confidence: confNum,
                predicted_class: data.variety,
                api_version: data.api_version || null,
                model_version: JSON.stringify({ pipeline: data.decision_rule || 'efficientnet_only' }),
                morphology_info: data.morphology_info || null,
                measurement_quality: data.measurement_quality || null,
                variety_characteristics: data.variety_characteristics || null
              }
            }
          }
        } catch {}
        if (out) {
          const noLeaf = dets.length === 0
          if (!noLeaf) {
            const retried = await autoPreprocessFromPreview()
            if (retried) return
          }
          const reasons = []
          if (noLeaf) reasons.push('Daun tidak terdeteksi')
          setResult({
            error: noLeaf ? 'Daun tidak terdeteksi pada gambar.' : 'Gambar di luar cakupan daun cabai (13 varietas) yang diteliti.',
            reasons,
            noLeaf
          })
        } else {
          try {
            if (data.preview_base64) {
              sessionStorage.setItem('last_preview', data.preview_base64)
              localStorage.setItem('last_preview', data.preview_base64)
            } else {
              const reader2 = new FileReader()
              reader2.onload = () => { try { sessionStorage.setItem('last_preview', reader2.result); localStorage.setItem('last_preview', reader2.result) } catch {} }
              reader2.readAsDataURL(blob)
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

  const autoPreprocessFromPreview = async () => {
    if (!previewUrl) return false
    return new Promise((resolve) => {
      const img = new Image()
      img.onload = () => {
        try {
          const w = img.naturalWidth, h = img.naturalHeight
          const canvas = document.createElement('canvas')
          canvas.width = w
          canvas.height = h
          const ctx = canvas.getContext('2d')
          ctx.drawImage(img, 0, 0)
          const data = ctx.getImageData(0, 0, w, h).data
          let minX = w, minY = h, maxX = 0, maxY = 0, count = 0
          for (let y = 0; y < h; y += 2) {
            for (let x = 0; x < w; x += 2) {
              const i = (y * w + x) * 4
              const r = data[i], g = data[i+1], b = data[i+2]
              const mx = Math.max(r, g, b)
              const mn = Math.min(r, g, b)
              const sat = mx - mn
              const isGreen = g > 90 && g > 1.12 * r && g > 1.12 * b && sat > 24
              if (isGreen) {
                count++
                if (x < minX) minX = x
                if (y < minY) minY = y
                if (x > maxX) maxX = x
                if (y > maxY) maxY = y
              }
            }
          }
          if (count < 500) return resolve(false)
          const bw = Math.max(1, maxX - minX)
          const bh = Math.max(1, maxY - minY)
          const cover = (bw * bh) / (w * h)
          if (cover > 0.85) return resolve(false)
          const m = 0.08
          const sx = Math.max(0, Math.floor(minX - bw * m))
          const sy = Math.max(0, Math.floor(minY - bh * m))
          const sw = Math.min(w - sx, Math.floor(bw * (1 + 2 * m)))
          const sh = Math.min(h - sy, Math.floor(bh * (1 + 2 * m)))
          const out = document.createElement('canvas')
          out.width = sw
          out.height = sh
          const octx = out.getContext('2d')
          octx.drawImage(img, sx, sy, sw, sh, 0, 0, sw, sh)
          out.toBlob(async (blob) => {
            if (!blob) return resolve(false)
            const cropped = new File([blob], 'snapshot_crop.jpg', { type: 'image/jpeg' })
            await onPredict(cropped)
            resolve(true)
          }, 'image/jpeg', 0.9)
        } catch {
          resolve(false)
        }
      }
      img.src = previewUrl
    })
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
            {Array.isArray(result.reasons) && result.reasons.length > 0 && (
              <ul>
                {result.reasons.map((r, i) => <li key={`r-${i}`}>{r}</li>)}
              </ul>
            )}
            {previewUrl && (
              <div style={{ position: 'relative', display: 'inline-block', marginTop: 8 }}
                onMouseDown={startCrop}
                onMouseMove={moveCrop}
                onMouseUp={endCrop}
              >
                <img id="live-preview" src={previewUrl} alt="Pratinjau" />
                {cropRect && (
                  <div style={{ position: 'absolute', left: cropRect.x, top: cropRect.y, width: cropRect.w, height: cropRect.h, border: '2px dashed #2F7E79', background: 'rgba(47,126,121,0.12)' }} />
                )}
              </div>
            )}
            
            <div className="actions" style={{ marginTop: 10 }}>
              <button className="btn" onClick={() => setResult(null)}>Ambil Ulang</button>
              {previewUrl && <button className="btn ghost" onClick={() => setCropRect({ x: 20, y: 20, w: 120, h: 120 })}>Mulai Crop</button>}
              {previewUrl && cropRect && <button className="btn outline" onClick={applyCropFromPreview}>Klasifikasi ROI</button>}
              {previewUrl && lastDetections?.length > 0 && <button className="btn outline" onClick={applyAutoROIFromPreview}>Auto ROI</button>}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
