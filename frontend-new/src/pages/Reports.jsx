import { useEffect, useMemo, useState } from 'react'
import { supabase } from '../lib/supabaseClient'
import { formatDate } from '../lib/date'

// Warna ikon seragam untuk semua varietas
const ICON_COLOR = '#2F7E79'

// 13 glyph unik untuk 13 varietas (sinkron dengan History)
const VARIETY_ORDER = [
  'Branang',
  'Carla_agrihorti',
  'Carvi_agrihorti',
  'Ciko',
  'Hot_beauty',
  'Hot_vision',
  'Inata_agrihorti',
  'Ivegri',
  'Leaf_Tanjung',
  'Lingga',
  'Mia',
  'Pertiwi',
  'Pilar',
]
const VARIETY_GLYPHS = ['●','■','▲','◆','▼','★','✦','⬟','⬢','⬣','⯁','⬠','⬡']
function glyphForVariety(name) {
  const key = String(name || '').trim()
  const idx = VARIETY_ORDER.indexOf(key)
  if (idx >= 0) return VARIETY_GLYPHS[idx]
  if (!key) return VARIETY_GLYPHS[0]
  let sum = 0
  for (let i = 0; i < key.length; i++) sum = (sum + key.charCodeAt(i)) >>> 0
  return VARIETY_GLYPHS[sum % VARIETY_GLYPHS.length]
}

// Util untuk format persen
function formatPercent(num) {
  if (typeof num !== 'number' || isNaN(num)) return '-'
  return `${(num * 100).toFixed(2)}%`
}

// Ambil nilai confidence numerik 0..1 dari item
function confidenceValue(item) {
  if (!item) return 0
  if (typeof item.confidence === 'number' && !isNaN(item.confidence)) return item.confidence
  const s = item.confidence_percentage
  if (typeof s === 'string') {
    const m = s.match(/([0-9]+(?:\.[0-9]+)?)/)
    if (m) {
      const v = parseFloat(m[1]) / 100
      return isNaN(v) ? 0 : Math.min(v, 0.9999)
    }
  }
  return 0
}

// Pemetaan tone: hijau (>=0.8), kuning (>=0.5), merah (<0.5), biru (default)
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

export default function Reports() {
  // Nama bulan Indonesia untuk header kalender
  const MONTH_NAMES_ID = ['Januari','Februari','Maret','April','Mei','Juni','Juli','Agustus','September','Oktober','November','Desember']
  const [items, setItems] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  // Filter
  const [filterVariety, setFilterVariety] = useState('')
  const [dateFrom, setDateFrom] = useState(null)
  const [dateTo, setDateTo] = useState(null)
  const [showDatePicker, setShowDatePicker] = useState(false)
  const [viewMonth, setViewMonth] = useState(new Date().getMonth() + 1)
  const [viewYear, setViewYear] = useState(new Date().getFullYear())
  const [selFrom, setSelFrom] = useState(null)
  const [selTo, setSelTo] = useState(null)
  const [showYearSelect, setShowYearSelect] = useState(false)
  // Admin-specific filters
  const [isAdmin, setIsAdmin] = useState(false)
  const [filterUserId, setFilterUserId] = useState('')
  // Pemetaan id->nama untuk tampilan admin
  const [profilesMap, setProfilesMap] = useState({})
  const userOptions = useMemo(() => {
    const ids = Array.from(new Set(items.map(it => it.user_id).filter(Boolean)))
    return ids.map(id => ({ id, label: (profilesMap[id]?.full_name || id) }))
      .sort((a, b) => a.label.localeCompare(b.label))
  }, [items, profilesMap])
  // Export options (admin)
  const [includeUserId, setIncludeUserId] = useState(true)
  const [includeApiModel, setIncludeApiModel] = useState(false)

  useEffect(() => {
    const load = async () => {
      setError('')
      setLoading(true)
      try {
        if (!supabase) {
          setItems([])
        } else {
          let uid = ''
          let admin = false
          try {
            const { data: u } = await supabase.auth.getUser()
            uid = u?.user?.id || ''
            if (uid) {
              const { data: prof } = await supabase.from('profiles').select('role').eq('id', uid).single()
              admin = String(prof?.role || '').trim().toLowerCase() === 'admin'
              setIsAdmin(admin)
            }
          } catch (_) {}
          let q = supabase.from('predictions').select('*').order('created_at', { ascending: false })
          if (!admin && uid) q = q.eq('user_id', uid)
          const { data, error } = await q
          if (error) throw error
          setItems(data || [])
          if (admin) {
            const ids = Array.from(new Set((data || []).map(it => it.user_id).filter(Boolean)))
            if (ids.length) {
              try {
                const { data: profs } = await supabase
                  .from('profiles')
                  .select('id, full_name, role')
                  .in('id', ids)
                const map = {}
                for (const p of (profs || [])) {
                  map[p.id] = { full_name: p.full_name || '', role: p.role || 'user' }
                }
                setProfilesMap(map)
              } catch (_) {}
            }
          }
        }
      } catch (e) {
        setError(e.message || 'Gagal memuat data laporan')
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  // Defaultkan opsi ekspor audit untuk admin
  useEffect(() => {
    if (isAdmin) {
      setIncludeApiModel(true)
    }
  }, [isAdmin])

  // Opsi dropdown berdasarkan data
  const staticVarieties = useMemo(() => [
    'Branang','Carla_agrihorti','Carvi_agrihorti','Ciko','Hot_beauty','Hot_vision','Inata_agrihorti','Ivegri','Leaf_Tanjung','Lingga','Mia','Pertiwi','Pilar'
  ], [])
  const dataVarieties = useMemo(() => Array.from(new Set(items.map((it) => String((it.predicted_class || it.variety || '')).trim()).filter(Boolean))), [items])
  const varietyOptions = useMemo(() => Array.from(new Set([ ...staticVarieties, ...dataVarieties ])).sort((a, b) => a.localeCompare(b)), [staticVarieties, dataVarieties])

  const startOfDay = (d) => new Date(d.getFullYear(), d.getMonth(), d.getDate(), 0, 0, 0, 0)
  const endOfDay = (d) => new Date(d.getFullYear(), d.getMonth(), d.getDate(), 23, 59, 59, 999)

  const filteredItems = useMemo(() => items.filter((it) => {
    const d = new Date(it.created_at)
    const v = String((it.predicted_class || it.variety || '')).trim()
    const vOk = !filterVariety || v === filterVariety
    const uOk = !filterUserId || it.user_id === filterUserId
    let rangeOk = true
    if (dateFrom && dateTo) {
      const t = d.getTime()
      rangeOk = t >= startOfDay(dateFrom).getTime() && t <= endOfDay(dateTo).getTime()
    } else if (dateFrom && !dateTo) {
      const t = d.getTime()
      rangeOk = t >= startOfDay(dateFrom).getTime() && t <= endOfDay(dateFrom).getTime()
    }
    return vOk && uOk && rangeOk
  }), [items, filterVariety, filterUserId, dateFrom, dateTo])

  // Ringkas nilai model_version (JSON) menjadi string ramah baca
function summarizeModelVersion(model_version) {
  try {
    const obj = typeof model_version === 'string' ? JSON.parse(model_version) : model_version
    if (!obj || typeof obj !== 'object') return ''
    const p = String(obj.pipeline || '').toLowerCase()
    if (p) {
      if (p.includes('meta')) return 'EfficientNet + XGBoost (Meta)'
      if (p.includes('xgboost_combined')) return 'EfficientNet + XGBoost (Combined)'
      if (p.includes('xgboost_original')) return 'EfficientNet + XGBoost (Original)'
      if (p.includes('efficientnet')) return 'EfficientNet Only'
      return obj.pipeline
    }
    const art = obj.artifacts || {}
    const yo = art.yolo || {}
    const ef = art.efficientnet || {}
    const xg = art.xgboost || {}
    const parts = []
    if (yo.file || yo.date) parts.push(`YOLO ${yo.file || '-'}${yo.date ? ` (${yo.date})` : ''}`)
    if (ef.file || ef.date) parts.push(`EfficientNet ${ef.file || '-'}${ef.date ? ` (${ef.date})` : ''}`)
    if (xg.file || xg.date) parts.push(`XGBoost ${xg.file || '-'}${xg.date ? ` (${xg.date})` : ''}`)
    return parts.length ? parts.join(' · ') : ''
  } catch (_) {
    return typeof model_version === 'string' ? model_version : ''
  }
}

  // Statistik ringkasan
  const totalItems = items.length
  const filteredCount = filteredItems.length
  // Rata-rata confidence (0-1) untuk item terfilter
  const avgConfidence = useMemo(() => {
    if (!filteredItems.length) return null
    const vals = filteredItems.map((it) => {
      if (typeof it.confidence === 'number' && !isNaN(it.confidence)) return it.confidence
      const s = it.confidence_percentage
      if (typeof s === 'string') {
        const m = s.match(/([0-9]+(?:\.[0-9]+)?)/)
        if (m) {
          const v = parseFloat(m[1]) / 100
          if (!isNaN(v)) return v
        }
      }
      return null
    }).filter((v) => typeof v === 'number')
    if (!vals.length) return null
    const sum = vals.reduce((a, b) => a + b, 0)
    return sum / vals.length
  }, [filteredItems])

  const formatDateRangeLabel = () => {
    if (dateFrom && dateTo) {
      const a = formatDate(dateFrom)
      const b = formatDate(dateTo)
      return `${a} — ${b}`
    }
    if (dateFrom && !dateTo) {
      return formatDate(dateFrom)
    }
    return null
  }

  const goPrevMonth = () => {
    if (viewMonth === 1) {
      setViewMonth(12)
      setViewYear(viewYear - 1)
    } else {
      setViewMonth(viewMonth - 1)
    }
  }
  const goNextMonth = () => {
    if (viewMonth === 12) {
      setViewMonth(1)
      setViewYear(viewYear + 1)
    } else {
      setViewMonth(viewMonth + 1)
    }
  }
  const onPickDate = (day) => {
    const picked = new Date(viewYear, viewMonth - 1, day)
    if (!selFrom) {
      setSelFrom(picked)
      setSelTo(null)
      return
    }
    if (selFrom && !selTo) {
      const from = startOfDay(selFrom)
      const to = startOfDay(picked)
      if (to < from) {
        setSelFrom(picked)
        setSelTo(null)
      } else {
        setSelTo(picked)
      }
      return
    }
    setSelFrom(picked)
    setSelTo(null)
  }

  const applyRange = () => {
    setDateFrom(selFrom || null)
    setDateTo(selTo || null)
    setShowDatePicker(false)
  }
  const clearRange = () => { setDateFrom(null); setDateTo(null); setSelFrom(null); setSelTo(null) }

  // Tahun untuk dropdown kalender (sekitar tahun berjalan)
  const currentYear = new Date().getFullYear()
  const calYearOptions = useMemo(() => Array.from({ length: 8 }, (_, i) => currentYear - 6 + i), [currentYear])

  const onDownloadCsv = () => {
    const showUserCol = isAdmin && includeUserId
    const headers = [
      'id','created_at',
      ...(showUserCol ? ['user'] : []),
      'filename','predicted_class','confidence','confidence_percentage',
      ...(includeApiModel ? ['api_version','model'] : []),
      // Kolom morfologi (nilai numerik tanpa satuan untuk tabel rapi)
      'panjang_daun_mm','lebar_daun_mm','keliling_daun_mm','panjang_tulang_daun_mm','rasio_bentuk_daun'
    ]
    const rows = filteredItems.map((it) => {
      const confPct = it.confidence_percentage || (typeof it.confidence === 'number' ? formatPercent(it.confidence) : '')
      const m = it.morphology_info || {}
      const base = [
        it.id,
        new Date(it.created_at).toISOString(),
      ]
      const userCol = showUserCol ? [profilesMap[it.user_id]?.full_name || it.user_id || ''] : []
      const common = [
        it.filename || '',
        it.predicted_class || '',
        typeof it.confidence === 'number' ? it.confidence : '',
        confPct,
      ]
      const apiModel = includeApiModel ? [it.api_version || '', summarizeModelVersion(it.model_version) || ''] : []
      const morph = [
        m.panjang_daun_mm ?? '',
        m.lebar_daun_mm ?? '',
        m.keliling_daun_mm ?? '',
        m.panjang_tulang_daun_mm ?? '',
        m.rasio_bentuk_daun != null ? Number(m.rasio_bentuk_daun).toFixed(2) : ''
      ]
      return [...base, ...userCol, ...common, ...apiModel, ...morph]
    })
    const csv = [headers.join(','), ...rows.map(r => r.map(v => {
      const s = String(v ?? '')
      if (s.includes(',') || s.includes('"') || s.includes('\n')) {
        return '"' + s.replace(/"/g, '""') + '"'
      }
      return s
    }).join(','))].join('\n')

    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    const namePart = [filterVariety || 'semua', dateFrom ? new Date(dateFrom).toISOString().slice(0,10) : '']
      .filter(Boolean).join('_')
    a.download = `unduh_prediksi_${namePart || 'terfilter'}.csv`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const onDownloadPdf = () => {
    // Buat layout sederhana untuk dicetak/diunduh sebagai PDF
    const w = window.open('', '_blank', 'width=900,height=1000')
    if (!w) return
    const rangeLabel = formatDateRangeLabel()
    const title = 'ChiliVar — Unduh Prediksi'
    const subtitle = 'Ringkasan dan daftar hasil sesuai filter'
    const styles = `
      body { font-family: Arial, sans-serif; color: #111827; padding: 24px; }
      h1 { font-size: 22px; margin: 0 0 4px; }
      .muted { color: #6b7280; }
      .chips { margin: 8px 0 16px; display: flex; gap: 8px; flex-wrap: wrap; }
      .chip { display: inline-block; padding: 4px 8px; border: 1px solid #e5e7eb; border-radius: 9999px; font-size: 12px; color: #374151; }
      .section { margin-top: 12px; }
      .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
      .card { border: 1px solid #e5e7eb; border-radius: 8px; padding: 10px; }
      .value { font-size: 18px; font-weight: 600; }
      .label { font-size: 12px; color: #6b7280; }
      table { width: 100%; border-collapse: collapse; margin-top: 12px; }
      th, td { border: 1px solid #e5e7eb; padding: 6px 8px; font-size: 12px; }
      th { background: #f9fafb; text-align: left; }
      @media print { .no-print { display: none; } }
    `
    const showUserCol = isAdmin && includeUserId
    const headers = [
      'Tanggal',
      ...(showUserCol ? ['User'] : []),
      'Filename','Varietas','Confidence (%)',
      ...(includeApiModel ? ['API','Model'] : []),
      'Panjang (mm)','Lebar (mm)','Keliling (mm)','Tulang (mm)','Rasio'
    ]
    const rows = filteredItems.map((it) => {
      const m = it.morphology_info || {}
      const conf = it.confidence_percentage || (typeof it.confidence === 'number' ? formatPercent(it.confidence) : '-')
      const base = [formatDate(it.created_at)]
      const userCol = showUserCol ? [profilesMap[it.user_id]?.full_name || it.user_id || '-'] : []
      const common = [it.filename || '-', it.predicted_class || '-', conf]
      const apiModel = includeApiModel ? [it.api_version || '-', summarizeModelVersion(it.model_version) || '-'] : []
      const morph = [
        m.panjang_daun_mm ?? '-',
        m.lebar_daun_mm ?? '-',
        m.keliling_daun_mm ?? '-',
        m.panjang_tulang_daun_mm ?? '-',
        m.rasio_bentuk_daun != null ? Number(m.rasio_bentuk_daun).toFixed(2) : '-'
      ]
      return [...base, ...userCol, ...common, ...apiModel, ...morph]
    })
    const tableHead = `<tr>${headers.map(h => `<th>${h}</th>`).join('')}</tr>`
    const tableRows = rows.map(r => `<tr>${r.map(c => `<td>${String(c)}</td>`).join('')}</tr>`).join('')
    const html = `
      <html>
        <head>
          <meta charset="utf-8" />
          <title>${title}</title>
          <style>${styles}</style>
        </head>
        <body>
          <h1>${title}</h1>
          <div class="muted">${subtitle}</div>
          <div class="chips">
            ${filterVariety ? `<span class="chip">Varietas: ${filterVariety}</span>` : ''}
            ${rangeLabel ? `<span class="chip">Tanggal: ${rangeLabel}</span>` : ''}
            <span class="chip">Item cocok: ${filteredItems.length}</span>
          </div>
          <div class="grid section">
            <div class="card"><div class="value">${totalItems}</div><div class="label">Total Riwayat</div></div>
            <div class="card"><div class="value">${filteredItems.length}</div><div class="label">Sesuai Filter</div></div>
            <div class="card"><div class="value">${avgConfidence != null ? (avgConfidence * 100).toFixed(2) + '%': '-'}</div><div class="label">Rata-rata Confidence</div></div>
          </div>
          <div class="section">
            <table>
              <thead>${tableHead}</thead>
              <tbody>${tableRows}</tbody>
            </table>
          </div>
          <div class="section no-print">
            <button onclick="window.print()">Cetak/Unduh PDF</button>
          </div>
        </body>
      </html>
    `
    w.document.write(html)
    w.document.close()
    w.focus()
  }

  return (
    <section className="card">
      <h1 className="card-title">Unduh Prediksi</h1>
      <p className="muted">Pilih filter untuk mengunduh data berdasarkan riwayat prediksi.</p>

      {loading && <div className="muted">Memuat…</div>}
      {error && <div className="error" role="alert">{error}</div>}

      {!loading && !error && (
        <>
        <div className="filters">
          <button className="filter-button" aria-label="Buka filter tanggal" onClick={() => { setSelFrom(dateFrom); setSelTo(dateTo); setShowDatePicker(v => !v) }}>Filter Tanggal</button>
          <select
            className="filter-select variety-select"
            value={filterVariety}
            onChange={(e) => setFilterVariety(e.target.value)}
            aria-label="Filter varietas"
            style={{ '--variety-ch': String(filterVariety || 'Semua Varietas').length }}
          >
            <option value="">Semua Varietas</option>
            {varietyOptions.map((v) => (
              <option key={`var-${v}`} value={v}>{v}</option>
            ))}
          </select>
          {isAdmin && (
            <select
              className="filter-select"
              value={filterUserId}
              onChange={(e) => setFilterUserId(e.target.value)}
              aria-label="Filter pengguna"
              style={{ minWidth: 220 }}
            >
              <option value="">Semua Pengguna</option>
              {userOptions.map(opt => (
                <option key={`uid-${opt.id}`} value={opt.id}>{opt.label}</option>
              ))}
            </select>
          )}
          <div className="spacer" />
          <button
            className="filter-button"
            onClick={() => {
              setFilterVariety('')
              setFilterUserId('')
              clearRange()
              setShowDatePicker(false)
            }}
          >Reset</button>
        </div>

        {/* Chips filter aktif */}
        {(filterVariety || formatDateRangeLabel()) && (
          <div className="filters" aria-label="Filter aktif" style={{ marginTop: 0 }}>
            {filterVariety && (
              <span className="filter-chip active">Varietas: {filterVariety}</span>
            )}
            {formatDateRangeLabel() && (
              <span className="filter-chip">Tanggal: {formatDateRangeLabel()}</span>
            )}
          </div>
        )}

        {/* Ringkasan statistik — satu background menyatu */}
        <div className="stats-pill" aria-label="Ringkasan riwayat dan filter">
          <div className="stat-item">
            <div className="stat-icon">
              <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="2" />
                <path d="M12 7v5l4 2" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </div>
            <div className="stat-value">{totalItems}</div>
            <div className="stat-label">Total Riwayat</div>
          </div>
          <div className="stat-item">
            <div className="stat-icon">
              <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M3 5h18l-7 8v6l-4-2v-4z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </div>
            <div className="stat-value">{filteredCount}</div>
            <div className="stat-label">Cocok dengan Filter</div>
          </div>
        </div>

        {showDatePicker && (
          <>
            <div className="filter-backdrop" onClick={() => setShowDatePicker(false)} aria-hidden="true" />
            <div className="filter-popover" role="dialog" aria-label="Filter tanggal">
              <div className="calendar">
                <div className="calendar-header">
                  <button className="nav" onClick={goPrevMonth} aria-label="Bulan sebelumnya">
                    <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M15 6l-6 6 6 6" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg>
                  </button>
                  {showYearSelect ? (
                    <select
                      className="year-select"
                      value={viewYear}
                      onChange={(e) => { setViewYear(Number(e.target.value)); setShowYearSelect(false) }}
                      onBlur={() => setShowYearSelect(false)}
                      onKeyDown={(e) => { if (e.key === 'Escape') setShowYearSelect(false) }}
                      aria-label="Pilih tahun"
                    >
                      {calYearOptions.map((y) => (
                        <option key={y} value={y}>{y}</option>
                      ))}
                    </select>
                  ) : (
                    <button className="btn small" onClick={() => setShowYearSelect(s => !s)} aria-expanded={showYearSelect} aria-controls="year-select">
                      {MONTH_NAMES_ID[viewMonth - 1]} {viewYear}
                    </button>
                  )}
                  <button className="nav" onClick={goNextMonth} aria-label="Bulan berikutnya">
                    <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 6l6 6-6 6" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg>
                  </button>
                </div>
                <div className="weekday-row">
                  <span>Min</span><span>Sen</span><span>Sel</span><span>Rab</span><span>Kam</span><span>Jum</span><span>Sab</span>
                </div>
                <div className="calendar-grid">
                  {Array.from({ length: new Date(viewYear, viewMonth - 1, 1).getDay() }).map((_, i) => (
                    <div key={`pad-${i}`} className="cell pad" />
                  ))}
                  {Array.from({ length: new Date(viewYear, viewMonth, 0).getDate() }).map((_, i) => {
                    const day = i + 1
                    const thisDate = new Date(viewYear, viewMonth - 1, day)
                    const isStart = !!selFrom && thisDate.toDateString() === selFrom?.toDateString()
                    const isEnd = !!selTo && thisDate.toDateString() === selTo?.toDateString()
                    let inRange = false
                    if (selFrom && selTo) {
                      inRange = thisDate >= startOfDay(selFrom) && thisDate <= endOfDay(selTo)
                    }
                    const cls = ["cell", "day"]
                    if (isStart) cls.push("start")
                    if (isEnd) cls.push("end")
                    if (inRange) cls.push("in-range")
                    return (
                      <button
                        key={`day-${day}`}
                        type="button"
                        className={cls.join(' ')}
                        onClick={() => onPickDate(day)}
                      >{day}</button>
                    )
                  })}
                </div>
                <div className="calendar-actions">
                  <button className="btn outline" onClick={clearRange}>Hapus</button>
                  <button className="btn" onClick={applyRange}>Terapkan</button>
                </div>
              </div>
            </div>
          </>
        )}

        <div className="actions" style={{ marginTop: 12 }}>
          <div className="small muted">Item cocok: {filteredItems.length}</div>
          <button className="filter-button" onClick={onDownloadCsv} disabled={filteredItems.length === 0}>Unduh CSV</button>
          <button className="filter-button" onClick={onDownloadPdf} disabled={filteredItems.length === 0}>Unduh PDF</button>
          {isAdmin && (
            <div style={{ marginTop: 8 }}>
              <span className="small muted">Opsi Ekspor:</span>
              <label style={{ marginLeft: 8 }}>
                <input type="checkbox" checked={includeUserId} onChange={(e) => setIncludeUserId(e.target.checked)} /> Sertakan User
              </label>
              <label style={{ marginLeft: 12 }} title="Menambahkan kolom versi API backend dan ringkasan model yang menghasilkan prediksi (berguna untuk audit/QA)">
                <input type="checkbox" checked={includeApiModel} onChange={(e) => setIncludeApiModel(e.target.checked)} /> Sertakan info API/Model (audit)
              </label>
            </div>
          )}
        </div>

        {/* (Panel Top Tidak Pasti dihapus sesuai permintaan) */}

        {/* Tinjauan singkat daftar terfilter (bergaya History) */}
        {filteredItems.length > 0 && (
          <div className="history-list" role="list" style={{ marginTop: 12 }}>
            {filteredItems.slice(0, 50).map((it) => {
              const confVal = confidenceValue(it)
              const tone = toneFromConfidence(confVal)
              const toneColor = TONE_COLOR[tone]
              const confPct = it.confidence_percentage || (typeof it.confidence === 'number' ? formatPercent(it.confidence) : '-')
              const confText = `Confidence: ${confPct}`
              return (
                <div key={it.id} role="listitem" className="history-item">
                  <div
                    className="history-icon"
                    aria-label={`ikon varietas ${(it.predicted_class || it.variety || '-')}`}
                    title={(it.predicted_class || it.variety || '')}
                    style={{ color: ICON_COLOR, fontSize: 18, lineHeight: '18px' }}
                  >
                    {glyphForVariety(it.predicted_class || it.variety)}
                  </div>
                  <div>
                    <div className="history-top">
                      <span className="history-pred">{it.predicted_class || it.variety || '-'}</span>
                      <span className="muted small">{confText}</span>
                    </div>
                    <div className="history-meta">
                      <span>{formatDate(it.created_at)}</span>
                      <span>•</span>
                      <span className="muted">{it.filename || '-'}</span>
                      {isAdmin && it.user_id && (
                        <>
                          <span>•</span>
                          <span className="muted">User: {profilesMap[it.user_id]?.full_name || it.user_id}</span>
                        </>
                      )}
                      {isAdmin && (
                        <>
                          <span>•</span>
                          <span className="muted">API: {it.api_version || '-'}</span>
                          <span>•</span>
                          <span className="muted">Model: {summarizeModelVersion(it.model_version) || '-'}</span>
                        </>
                      )}
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        )}
        </>
      )}
    </section>
  )
}
