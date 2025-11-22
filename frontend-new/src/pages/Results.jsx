import { useEffect, useMemo, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'

function formatPercent(num) {
  if (typeof num !== 'number' || isNaN(num)) return '-'
  return `${(num * 100).toFixed(2)}%`
}

function confidenceDisplay(r) {
  if (!r) return '-'
  return r.confidence_percentage || (typeof r.confidence === 'number' ? formatPercent(r.confidence) : '-')
}

function confidenceValue(r) {
  // Numeric value in range [0,1] for progress bar
  if (!r) return 0
  if (typeof r.confidence === 'number' && !isNaN(r.confidence)) return r.confidence
  const s = r.confidence_percentage
  if (typeof s === 'string') {
    const m = s.match(/([0-9]+(?:\.[0-9]+)?)/)
    if (m) {
      const v = parseFloat(m[1]) / 100
      return isNaN(v) ? 0 : Math.min(v, 0.9999)
    }
  }
  return 0
}

// Aturan interpretasi spesifik (selaras dengan backend)
function isSpecialVariety(name) {
  if (!name) return false
  const n = String(name).toLowerCase()
  return n === 'ciko' || n === 'branang' || n === 'mia'
}

function deriveUncertain(result) {
  if (!result) return false
  const conf = confidenceValue(result)
  const tone = toneFromConfidence(conf)
  const backendFlag = !!result.is_uncertain
  return backendFlag || tone === 'yellow' || tone === 'blue'
}

function confidenceLevel(result) {
  const v = confidenceValue(result)
  if (v >= 0.95) return 'Sangat yakin'
  if (v >= 0.90) return 'Cukup yakin'
  return 'Tidak pasti'
}

function isOutOfScope(result) {
  if (!result) return false
  if (result.is_out_of_scope) return true
  const name = String(result.variety || '').trim().toLowerCase()
  return name === 'unknown'
}

// Pemetaan tone untuk pewarnaan ringkasan hasil
function toneFromConfidence(v) {
  if (typeof v !== 'number' || isNaN(v) || v === 0) return 'blue'
  if (v < 0.7) return 'yellow'
  return 'green'
}

const TONE_COLOR = {
  green: '#2F7E79',
  yellow: '#F4C945',
  red: '#E34D4D',
  blue: '#7A8FB9',
}

function listIssues(measurement_quality) {
  try {
    const issues = measurement_quality?.issues
    if (Array.isArray(issues)) return issues.filter(Boolean)
    return []
  } catch {
    return []
  }
}

function buildCsv(result) {
  const r = result || {}
  const m = r.morphology_info || {}
  const headers = [
    'variety', 'confidence_raw', 'confidence_percentage', 'decision_rule',
    'panjang_daun_mm', 'lebar_daun_mm', 'keliling_daun_mm', 'panjang_tulang_daun_mm', 'rasio_bentuk_daun',
    'scale_mm_per_px'
  ]
  const row = [
    r.variety,
    r.confidence_raw,
    r.confidence_percentage || (typeof r.confidence === 'number' ? formatPercent(r.confidence) : ''),
    r.decision_rule,
    m.panjang_daun_mm,
    m.lebar_daun_mm,
    m.keliling_daun_mm,
    m.panjang_tulang_daun_mm,
    m.rasio_bentuk_daun != null ? Number(m.rasio_bentuk_daun).toFixed(2) : '',
    m.scale_mm_per_px
  ]
  const escape = (v) => {
    if (v == null) return ''
    const s = String(v)
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}` : s
  }
  const csv = [headers.join(','), row.map(escape).join(',')].join('\n')
  return csv
}

export default function Results() {
  const location = useLocation()
  const navigate = useNavigate()
  const [copied, setCopied] = useState(false)
  const input = location.state?.result || null
  const result = useMemo(() => {
    if (input) return input
    try {
      const raw = sessionStorage.getItem('last_result') || localStorage.getItem('last_result')
      return raw ? JSON.parse(raw) : null
    } catch {
      return null
    }
  }, [input])
  const previewUrlInitial = (() => {
    const fromState = location.state?.previewUrl || null
    if (fromState) return fromState
    try {
      const s1 = sessionStorage.getItem('last_preview')
      if (s1) return s1
      const s2 = localStorage.getItem('last_preview')
      return s2 || null
    } catch {
      return null
    }
  })()
  const [imageSrc, setImageSrc] = useState(previewUrlInitial)
  useEffect(() => {
    if (!imageSrc) {
      try {
        const s1 = sessionStorage.getItem('last_preview')
        if (s1) { setImageSrc(s1); return }
        const s2 = localStorage.getItem('last_preview')
        if (s2) setImageSrc(s2)
      } catch {}
    }
  }, [imageSrc])
  useEffect(() => {
    if (!imageSrc && result?.preview_base64) {
      setImageSrc(result.preview_base64)
    }
  }, [imageSrc, result])

  

  useEffect(() => {
    // Jika tidak ada hasil, tetap render penjelasan placeholder
  }, [result])

  const onBack = () => navigate(-1)

  const onDownloadCSV = () => {
    if (!result) return
    const csv = buildCsv(result)
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `hasil_klasifikasi_${Date.now()}.csv`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const onDownloadPDF = () => {
    if (!result) return
    // Cetak ke PDF via jendela baru agar layout ringkas
    const w = window.open('', '_blank', 'width=800,height=900')
    if (!w) return
    const m = result.morphology_info || {}
    const vc = result.variety_characteristics || {}
    const uncertain = deriveUncertain(result)
    const level = confidenceLevel(result)
    const issues = []
    const qualityStatus = null
    const out = isOutOfScope(result)
    const html = `
      <html>
        <head>
          <meta charset="utf-8" />
          <title>Hasil Klasifikasi</title>
          <style>
            body { font-family: Arial, sans-serif; color: #111827; padding: 24px; }
            h1 { font-size: 22px; margin: 0 0 10px; }
            h2 { font-size: 18px; margin: 16px 0 8px; }
            .muted { color: #6b7280; }
            .section { margin-top: 10px; border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; }
            .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
            .row { display: grid; grid-template-columns: 180px 1fr; gap: 8px; }
            .label { color: #6b7280; }
          </style>
        </head>
        <body>
          <h1>Hasil Klasifikasi</h1>
          <div class="muted">Ringkasan hasil klasifikasi varietas daun cabai</div>
          <div class="section">
            <div class="row"><div class="label">Varietas</div><div><strong>${result.variety || '-'}</strong></div></div>
            <div class="row"><div class="label">Confidence</div><div>${confidenceDisplay(result)} (${level})${out ? ' · <span class="muted">Tidak berlaku untuk input di luar cakupan</span>' : ''}</div></div>
            <div class="row"><div class="label">Status</div><div>${uncertain ? 'Tidak Pasti' : 'Cukup Pasti'}</div></div>
          </div>
          <div class="grid">
            <div class="section">
              <h2>Fitur Morfologi</h2>
              ${out ? '<div class="muted">Di luar cakupan daun cabai (13 varietas). Morfologi tidak ditampilkan.</div>' : `
              <div class="row"><div class="label">Panjang</div><div>${fmtMm(m.panjang_daun_mm)}</div></div>
              <div class="row"><div class="label">Lebar</div><div>${fmtMm(m.lebar_daun_mm)}</div></div>
              <div class="row"><div class="label">Keliling</div><div>${fmtMm(m.keliling_daun_mm)}</div></div>
              <div class="row"><div class="label">Panjang Tulang</div><div>${fmtMm(m.panjang_tulang_daun_mm)}</div></div>
              <div class="row"><div class="label">Rasio Bentuk</div><div>${m.rasio_bentuk_daun != null ? Number(m.rasio_bentuk_daun).toFixed(2) : '-'}</div></div>
              `}
            </div>
            <div class="section">
              <h2>Karakteristik Varietas</h2>
              <div class="row"><div class="label">Nama</div><div>${vc.name || '-'}</div></div>
              <div class="row"><div class="label">Bentuk</div><div>${vc.shape || '-'}</div></div>
              <div class="row"><div class="label">Rentang Panjang</div><div>${vc.length_range || '-'}</div></div>
              <div class="row"><div class="label">Rentang Lebar</div><div>${vc.width_range || '-'}</div></div>
            </div>
          </div>
          
          <script>window.print(); setTimeout(() => window.close(), 500);</script>
        </body>
      </html>
    `
    w.document.write(html)
    w.document.close()
  }

  const onShare = async () => {
    if (!result) return
    const uncertain = deriveUncertain(result)
    const level = confidenceLevel(result)
    const summaryText = `Varietas: ${result.variety}\nConfidence: ${confidenceDisplay(result)} (${level})\nStatus: ${uncertain ? 'Tidak Pasti' : 'Cukup Pasti'}`
    try {
      if (navigator.share) {
        await navigator.share({ title: 'Hasil Klasifikasi', text: summaryText })
        return
      }
      await navigator.clipboard.writeText(summaryText)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      setCopied(false)
    }
  }

  const m = result?.morphology_info || {}
  const mq = result?.measurement_quality || null
  const vcRaw = result?.variety_characteristics || {}
  const fmtMm = (v) => {
    if (v == null) return '-'
    const num = Number(v)
    if (isNaN(num)) return '-'
    if (num >= 100) return `${(num/10).toFixed(1)} cm`
    return `${Math.round(num)} mm`
  }
  const cardGreen = { bg: '#2A7F65', border: '#1D5E4D', text: '#ffffff', muted: '#FFFFFFCC' }
  const cardStyle = { background: cardGreen.bg, border: `1px solid ${cardGreen.border}`, borderRadius: 8, padding: '8px 10px', color: cardGreen.text }
  const mutedStyle = { color: cardGreen.muted }
  const toRange = (stat) => {
    if (!stat || typeof stat !== 'object') return '-'
    const mn = stat.min, mx = stat.max
    if (mn == null || mx == null) return '-'
    const a = Math.round(Number(mn))
    const b = Math.round(Number(mx))
    return `${a}-${b} mm`
  }
  const vcDisplay = (raw) => {
    if (!raw || typeof raw !== 'object' || Object.keys(raw).length === 0) return {}
    const ratio = Number(m?.rasio_bentuk_daun)
    const shape = (isNaN(ratio) ? '-' : (ratio >= 0.85 ? 'bulat' : (ratio >= 0.65 ? 'oval' : 'lonjong')))
    return {
      name: result?.variety,
      shape,
      length_range: toRange(raw.panjang_daun_mm),
      width_range: toRange(raw.lebar_daun_mm),
    }
  }
  const vcFallbackMap = {
    Branang: { name: 'Branang', shape: 'lonjong hingga oval', length_range: '35-55 mm', width_range: '15-40 mm' },
    Carla_agrihorti: { name: 'Carla_agrihorti', shape: 'bulat telur', length_range: '40-55 mm', width_range: '18-40 mm' },
    Carvi_agrihorti: { name: 'Carvi_agrihorti', shape: 'bulat telur hingga oval', length_range: '40-60 mm', width_range: '25-40 mm' },
    Ciko: { name: 'Ciko', shape: 'lonjong', length_range: '40-55 mm', width_range: '20-45 mm' },
    Hot_beauty: { name: 'Hot_beauty', shape: 'lonjong hingga oval', length_range: '35-55 mm', width_range: '18-40 mm' },
    Hot_vision: { name: 'Hot_vision', shape: 'bulat telur', length_range: '40-60 mm', width_range: '25-40 mm' },
    Inata_agrihorti: { name: 'Inata_agrihorti', shape: 'oval', length_range: '35-55 mm', width_range: '18-40 mm' },
    Ivegri: { name: 'Ivegri', shape: 'bulat telur', length_range: '45-60 mm', width_range: '25-45 mm' },
    Leaf_Tanjung: { name: 'Leaf_Tanjung', shape: 'lonjong hingga bulat', length_range: '35-55 mm', width_range: '15-45 mm' },
    Lingga: { name: 'Lingga', shape: 'bulat telur', length_range: '40-60 mm', width_range: '20-40 mm' },
    Mia: { name: 'Mia', shape: 'bulat telur', length_range: '40-55 mm', width_range: '25-40 mm' },
    Pertiwi: { name: 'Pertiwi', shape: 'bulat telur', length_range: '40-60 mm', width_range: '25-45 mm' },
    Pilar: { name: 'Pilar', shape: 'bulat telur hingga oval', length_range: '35-55 mm', width_range: '20-40 mm' }
  }
  const vcEmpty = vcRaw && typeof vcRaw === 'object' && Object.keys(vcRaw).length === 0
  const vc = vcEmpty ? (vcFallbackMap[result?.variety] || {}) : vcDisplay(vcRaw)
  const uncertain = result ? deriveUncertain(result) : false
  const level = result ? confidenceLevel(result) : '-'
  const issues = result ? listIssues(result?.measurement_quality) : []
  const qualityStatus = result?.measurement_quality?.status || null
  const tone = result ? toneFromConfidence(confidenceValue(result)) : 'blue'
  const toneColor = TONE_COLOR[tone]
  const out = result ? isOutOfScope(result) : false

  return (
    <section className="card">
      <h1 className="card-title">Hasil Klasifikasi</h1>
      {!result ? (
        <>
          <p className="muted">Tidak ada data untuk ditampilkan. Silakan lakukan Upload terlebih dahulu untuk melihat hasil klasifikasi.</p>
          <div className="actions">
            <button className="btn" onClick={() => navigate('/upload')}>Ke Upload</button>
          </div>
        </>
      ) : (
        <>
          <div className="results-grid">
            <div className="results-left">
              {imageSrc && (
                <div className="preview"><img src={imageSrc} alt="preview hasil" loading="lazy" decoding="async" /></div>
              )}
            </div>
            <div className="results-right">
              <div className="section" style={{ padding: '10px 12px 0', margin: '12px 0 -14px 0', border: '0.75px solid #e5e7eb' }}>
                <div className={`summary tone-${tone}`} style={{ borderLeft: `6px solid ${cardGreen.border}`, background: '#EAF6F2', borderRadius: 12, padding: '10px 12px', marginBottom: 18 }}>
                  <div className="summary-top">
                    <div className="summary-title">
                      <span className="muted">Varietas</span>
                      <div className="summary-name">{result.variety}</div>
                      {isOutOfScope(result) && (
                        <div className="small muted" style={{ marginTop: 4 }}>Di luar cakupan daun cabai (13 varietas)</div>
                      )}
                    </div>
                  </div>
                  <div className="summary-confidence">Confidence: {confidenceDisplay(result)}</div>
                  <div className="progress" style={{ height: 12, borderRadius: 9999, background: '#D6EDE6' }}>
                    <div className="progress-bar" style={{ width: `${confidenceValue(result) * 100}%`, background: cardGreen.bg, borderRadius: 9999 }} />
                  </div>
                  <div className="meta"></div>
                </div>
                <div className="section-title" style={{ margin: '18px 0 12px' }}>Fitur Kunci</div>
                {out ? (
                  <p className="muted">Di luar cakupan daun cabai (13 varietas). Morfologi tidak ditampilkan.</p>
                ) : (
                  <div className="metrics" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, max-content))', gap: 4, marginBottom: 4, justifyContent: 'space-evenly' }}>
                    <div className="metric" style={cardStyle}><span className="muted" style={mutedStyle}>Panjang</span><strong>{fmtMm(m.panjang_daun_mm)}</strong></div>
                    <div className="metric" style={cardStyle}><span className="muted" style={mutedStyle}>Lebar</span><strong>{fmtMm(m.lebar_daun_mm)}</strong></div>
                    <div className="metric" style={cardStyle}><span className="muted" style={mutedStyle}>Keliling</span><strong>{fmtMm(m.keliling_daun_mm)}</strong></div>
                    <div className="metric" style={cardStyle}><span className="muted" style={mutedStyle}>Tulang</span><strong>{fmtMm(m.panjang_tulang_daun_mm)}</strong></div>
                    <div className="metric" style={cardStyle}><span className="muted" style={mutedStyle}>Rasio</span><strong>{m.rasio_bentuk_daun != null ? Number(m.rasio_bentuk_daun).toFixed(2) : '-'}</strong></div>
                  </div>
                )}
                <div className="section-title" style={{ margin: '28px 0 12px' }}>Karakteristik Varietas</div>
                {vc && Object.keys(vc).length > 0 ? (
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, justifyContent: 'space-evenly', width: '100%' }}>
                    <div style={{ ...cardStyle, minWidth: 140 }}>
                      <div className="muted" style={{ fontSize: 12, ...mutedStyle }}>Varietas</div>
                      <div style={{ fontWeight: 600 }}>{vc.name || '-'}</div>
                    </div>
                    <div style={{ ...cardStyle, minWidth: 140 }}>
                      <div className="muted" style={{ fontSize: 12, ...mutedStyle }}>Bentuk</div>
                      <div style={{ fontWeight: 600 }}>{vc.shape || '-'}</div>
                    </div>
                    <div style={{ ...cardStyle, minWidth: 160 }}>
                      <div className="muted" style={{ fontSize: 12, ...mutedStyle }}>Rentang Panjang</div>
                      <div style={{ fontWeight: 600 }}>{vc.length_range || '-'}</div>
                    </div>
                    <div style={{ ...cardStyle, minWidth: 160 }}>
                      <div className="muted" style={{ fontSize: 12, ...mutedStyle }}>Rentang Lebar</div>
                      <div style={{ fontWeight: 600 }}>{vc.width_range || '-'}</div>
                    </div>
                  </div>
                ) : (
                  <div className="muted">Tidak ada karakteristik tersedia.</div>
                )}
              </div>
              {uncertain && (
                <div className="section" style={{ marginTop: 12 }}>
                  <div className="section-title">Peringatan Akurasi</div>
                  <p className="error">Confidence rendah atau pengukuran kurang stabil. Disarankan foto ulang agar hasil lebih akurat.</p>
                  {mq && Array.isArray(mq.issues) && mq.issues.length > 0 && (
                    <div className="small muted" style={{ marginTop: 6 }}>Catatan: {mq.issues.join(', ')}</div>
                  )}
                  <div className="actions" style={{ marginTop: 10 }}>
                    <button className="btn" onClick={() => navigate('/upload')}>Ambil Ulang</button>
                  </div>
                </div>
              )}
              
            </div>
          </div>
          <div className="actions">
            <button className="btn outline" onClick={onBack}>Kembali</button>
            <button className="btn outline" onClick={onShare}>{copied ? 'Disalin!' : 'Bagikan'}</button>
          </div>
        </>
      )}
    </section>
  )
}
