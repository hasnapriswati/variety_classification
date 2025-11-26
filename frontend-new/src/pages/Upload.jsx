import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { supabase } from '../lib/supabaseClient'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000'

// Batasan ukuran & kompresi
const MAX_FILE_SIZE = 4 * 1024 * 1024 // 4MB
const MAX_DIMENSION = 1600 // px
const JPEG_QUALITY = 0.85
const WEBP_QUALITY = 0.8

// Kompresi ke format yang ditentukan (default WebP)
async function compressImage(file, format = 'webp') {
  return new Promise((resolve, reject) => {
    try {
      const url = URL.createObjectURL(file)
      const img = new Image()
      img.onload = () => {
        const { width, height } = img
        const maxSide = Math.max(width, height)
        const scale = maxSide > MAX_DIMENSION ? (MAX_DIMENSION / maxSide) : 1
        const targetW = Math.round(width * scale)
        const targetH = Math.round(height * scale)
        const canvas = document.createElement('canvas')
        canvas.width = targetW
        canvas.height = targetH
        const ctx = canvas.getContext('2d')
        ctx.drawImage(img, 0, 0, targetW, targetH)
        const mime = format === 'jpeg' ? 'image/jpeg' : 'image/webp'
        const quality = format === 'jpeg' ? JPEG_QUALITY : WEBP_QUALITY
        canvas.toBlob((blob) => {
          if (!blob) return reject(new Error('Gagal mengompresi gambar'))
          const ext = format === 'jpeg' ? '.jpg' : '.webp'
          const out = new File([blob], file.name.replace(/\.(png|jpg|jpeg|webp)$/i, ext), { type: mime })
          resolve({ file: out, blob, width: targetW, height: targetH })
          URL.revokeObjectURL(url)
        }, mime, quality)
      }
      img.onerror = () => reject(new Error('Gagal memuat gambar untuk kompresi'))
      img.src = url
    } catch (e) { reject(e) }
  })
}

export default function Upload() {
  const navigate = useNavigate()
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  
  const dropRef = useRef(null)
  const fileInputRef = useRef(null)

  useEffect(() => {
    const preventDefaults = (e) => { e.preventDefault(); e.stopPropagation() }
    const dropArea = dropRef.current
    const highlight = () => dropArea.classList.add('drop--highlight')
    const unhighlight = () => dropArea.classList.remove('drop--highlight')
    ;['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
      dropArea.addEventListener(eventName, preventDefaults, false)
    })
    ;['dragenter', 'dragover'].forEach(e => dropArea.addEventListener(e, highlight))
    ;['dragleave', 'drop'].forEach(e => dropArea.addEventListener(e, unhighlight))
    dropArea.addEventListener('drop', (e) => {
      const dt = e.dataTransfer
      const files = dt.files
      if (files && files[0]) onFileSelected(files[0])
    })
    return () => {
      ;['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.removeEventListener(eventName, preventDefaults)
      })
    }
  }, [])

  const [sizeNote, setSizeNote] = useState('')

  const onFileSelected = async (f) => {
    if (!f) return
    setSizeNote('')
    try {
      if (f.size > MAX_FILE_SIZE) {
        const { file: compressed, blob } = await compressImage(f, 'jpeg')
        setFile(compressed)
        setPreviewUrl(URL.createObjectURL(blob))
        try {
          const reader = new FileReader()
          reader.onload = () => { try { sessionStorage.setItem('last_preview', reader.result); localStorage.setItem('last_preview', reader.result) } catch {} }
          reader.readAsDataURL(blob)
        } catch {}
        setSizeNote('Gambar terlalu besar, dikompresi otomatis agar lebih ringan.')
      } else {
        setFile(f)
        setPreviewUrl(URL.createObjectURL(f))
        try {
          const reader = new FileReader()
          reader.onload = () => { try { sessionStorage.setItem('last_preview', reader.result); localStorage.setItem('last_preview', reader.result) } catch {} }
          reader.readAsDataURL(f)
        } catch {}
      }
    } catch (e) {
      setFile(f)
      setPreviewUrl(URL.createObjectURL(f))
      try {
        const reader = new FileReader()
        reader.onload = () => { try { sessionStorage.setItem('last_preview', reader.result); localStorage.setItem('last_preview', reader.result) } catch {} }
        reader.readAsDataURL(f)
      } catch {}
      setSizeNote('Gagal mengompresi, menggunakan gambar asli.')
    }
  }

  const onPredict = async (customFile = null) => {
    const targetFile = customFile || file
    if (!targetFile) return
    
    setLoading(true)
    setResult(null)
    
    const formData = new FormData()
    formData.append('file', targetFile)
    
    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      })
      
      if (!response.ok) {
        const err = await response.json().catch(() => ({}))
        throw new Error(err.error || 'Gagal melakukan prediksi')
      }
      
      const data = await response.json()
      
      setLoading(false)
      if (data && data.success) {
        const dets = Array.isArray(data?.yolo?.detections) ? data.yolo.detections : []
        const maxDetConf = dets.length ? Math.max(...dets.map(d => Number(d?.confidence || 0))) : 0
        const hasLeaf = dets.length > 0
        const qualityStatus = data?.measurement_quality?.status || 'ok'
        const confNumRaw = typeof data.confidence === 'number' && !isNaN(data.confidence) ? data.confidence : 0
        const out = (!!data.is_out_of_scope)
          || (String(data.variety || '').trim().toLowerCase() === 'unknown')
          || (!hasLeaf)
        try {
          const raw = JSON.stringify(data)
          sessionStorage.setItem('last_result', raw)
          localStorage.setItem('last_result', raw)
          if (data.preview_base64) {
            sessionStorage.setItem('last_preview', data.preview_base64)
            localStorage.setItem('last_preview', data.preview_base64)
          } else {
            const reader = new FileReader()
            reader.onload = () => { try { sessionStorage.setItem('last_preview', reader.result); localStorage.setItem('last_preview', reader.result) } catch {} }
            reader.readAsDataURL(targetFile)
          }
        } catch {}
        if (out) {
          const noLeaf = dets.length === 0
          if (!noLeaf) {
            const retried = await autoPreprocessAndPredict()
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
            const { data: sessionData } = await supabase.auth.getSession()
            const user = sessionData?.session?.user
            if (user) {
              const confNum = (typeof data.confidence === 'number' && !isNaN(data.confidence))
                ? data.confidence
                : (typeof data.confidence_percentage === 'string'
                    ? (parseFloat((data.confidence_percentage.match(/([0-9]+(?:\.[0-9]+)?)/)?.[1] || '0')) / 100)
                    : 0)

              const base = {
                user_id: user.id,
                confidence: confNum,
                confidence_percentage: data.confidence_percentage || null,
                filename: targetFile?.name || ''
              }
              const extra = {
                api_version: data.api_version || null,
                model_version: JSON.stringify({ pipeline: data.decision_rule || 'efficientnet_only' })
              }
              const essentials = {
                user_id: base.user_id,
                filename: base.filename,
                confidence: base.confidence
              }
              let payload = {
                ...base,
                predicted_class: data.variety,
                ...extra,
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
                  ...essentials,
                  predicted_class: data.variety,
                  api_version: data.api_version || null,
                  model_version: JSON.stringify({ pipeline: data.decision_rule || 'efficientnet_only' }),
                  morphology_info: data.morphology_info || null,
                  measurement_quality: data.measurement_quality || null,
                  variety_characteristics: data.variety_characteristics || null
                }
              }
              if (error) console.error('Error saving prediction:', error)
            }
          } catch (_) {}
          navigate('/results', { state: { result: data, previewUrl } })
        }
      } else {
        setResult({ error: data?.error || 'Prediksi gagal' })
      }
    } catch (error) {
      setResult({ error: error.message })
      setLoading(false)
    }
  }

  const handleFileChange = (e) => {
    const f = e.target.files?.[0]
    if (f) onFileSelected(f)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    const f = e.dataTransfer.files?.[0]
    if (f) onFileSelected(f)
  }

  const clearFile = () => {
    setFile(null)
    setPreviewUrl('')
    setResult(null)
    setSizeNote('')
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const autoPreprocessAndPredict = async () => {
    if (!previewUrl || !file) return false
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
            const cropped = new File([blob], (file?.name || 'crop.jpg').replace(/\.(png|jpg|jpeg|webp)$/i, '.jpg'), { type: 'image/jpeg' })
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
        <h1 className="card-title">Upload Gambar</h1>
        <p className="muted">Pilih atau drag & drop gambar untuk mengidentifikasi varietas kentang</p>

        <div
          ref={dropRef}
          className="drop"
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
        >
          <div>
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
              <polyline points="17 8 12 3 7 8"></polyline>
              <line x1="12" y1="3" x2="12" y2="15"></line>
            </svg>
            <h3 className="card-title center">Upload Gambar</h3>
            <p className="muted">Drag & drop gambar di sini atau klik untuk memilih</p>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              id="file-input"
              style={{ display: 'none' }}
            />
            <label htmlFor="file-input" className="btn">Pilih Gambar</label>
            {sizeNote && <p className="small info" style={{ marginTop: 8 }}>{sizeNote}</p>}
          </div>
        </div>

        {file && (
          <div className="preview">
            <img id="upload-preview" src={previewUrl} alt="Preview" />
            <div style={{ marginTop: 10 }}>
              <p className="small"><strong>File:</strong> {file.name}</p>
              <p className="small"><strong>Ukuran:</strong> {(file.size / 1024).toFixed(2)} KB</p>
            </div>
            <div className="actions-row" style={{ justifyContent: 'center', marginTop: 14 }}>
              <button onClick={clearFile} className="btn outline">Hapus</button>
              <button 
                onClick={() => onPredict()} 
                disabled={loading}
                className="btn"
              >
                {loading ? 'Memproses...' : 'Prediksi'}
              </button>
              
            </div>
          </div>
        )}

        {result?.error && (
          <div className="section" style={{ marginTop: 12 }}>
            <div className="section-title">Klasifikasi tidak dapat dilakukan</div>
            <p className="error" role="alert">{result.error}</p>
            {result.noLeaf && (
              <div className="muted small">Pastikan gambar berisi daun cabai yang jelas, tidak terlalu gelap/terang, dan memenuhi area gambar.</div>
            )}
            {Array.isArray(result.reasons) && result.reasons.length > 0 && (
              <ul>
                {result.reasons.map((r, i) => <li key={`r-${i}`}>{r}</li>)}
              </ul>
            )}
            
            
            
          </div>
        )}

        
      </div>
    </div>
  )
}
