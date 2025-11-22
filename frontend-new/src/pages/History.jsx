import { useEffect, useState } from 'react'
import { supabase } from '../lib/supabaseClient'
import { formatDate } from '../lib/date'

// Ikon bentuk untuk penanda tiap item riwayat
function ShapeIcon({ shape = 'circle', color = '#22c55e', size = 18 }) {
  const common = { width: size, height: size, viewBox: '0 0 16 16', "aria-hidden": true }
  if (shape === 'square') {
    return (
      <svg {...common}>
        <rect x="2" y="2" width="12" height="12" rx="2" fill={color} />
      </svg>
    )
  }
  if (shape === 'diamond') {
    return (
      <svg {...common}>
        <rect x="2" y="2" width="12" height="12" transform="rotate(45 8 8)" fill={color} />
      </svg>
    )
  }
  if (shape === 'triangle') {
    return (
      <svg {...common}>
        <polygon points="8,2 14,14 2,14" fill={color} />
      </svg>
    )
  }
  // default: circle
  return (
    <svg {...common}>
      <circle cx="8" cy="8" r="6" fill={color} />
    </svg>
  )
}
// Warna ikon seragam untuk semua varietas
const ICON_COLOR = '#2F7E79'

// 13 glyph unik untuk 13 varietas (warna seragam, bentuk berbeda)
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
  // Fallback deterministik bila label tidak dikenal
  let sum = 0
  for (let i = 0; i < key.length; i++) sum = (sum + key.charCodeAt(i)) >>> 0
  return VARIETY_GLYPHS[sum % VARIETY_GLYPHS.length]
}

function getVisual(label) {
  // Bentuk tetap bervariasi berdasarkan nama agar ikonnya tidak monoton
  const base = typeof label === 'string' ? label : ''
  const shapes = ['circle', 'square', 'diamond', 'triangle']
  const sum = base.split('').reduce((a, c) => a + c.charCodeAt(0), 0)
  const shape = shapes[sum % shapes.length]
  return { shape }
}

// Mapping warna mengikuti palet pada gambar (Bilpay-like)
const TONE_COLOR = {
  green: '#2F7E79',   // hijau kartu
  yellow: '#F4C945',  // kuning emas
  red: '#E34D4D',     // merah kartu
  blue: '#7A8FB9'     // biru netral
}

// Ikon+warna konsisten per-varietas (opsi A)
const VARIETY_VISUALS = {
  Branang:            { shape: 'circle',   color: '#2F7E79' },
  Carla_agrihorti:    { shape: 'square',   color: '#F4C945' },
  Carvi_agrihorti:    { shape: 'diamond',  color: '#7A8FB9' },
  Ciko:               { shape: 'triangle', color: '#E34D4D' },
  Hot_beauty:         { shape: 'circle',   color: '#3A948C' },
  Hot_vision:         { shape: 'square',   color: '#EABF3B' },
  Inata_agrihorti:    { shape: 'diamond',  color: '#5B7FBF' },
  Ivegri:             { shape: 'triangle', color: '#C53A3A' },
  Leaf_Tanjung:       { shape: 'circle',   color: '#21564E' },
  Lingga:             { shape: 'square',   color: '#D5AF2E' },
  Mia:                { shape: 'diamond',  color: '#3B82F6' },
  Pertiwi:            { shape: 'triangle', color: '#EF4444' },
  Pilar:              { shape: 'circle',   color: '#84cc16' }
}

function varietyVisual(name) {
  const key = typeof name === 'string' ? name.trim() : ''
  if (VARIETY_VISUALS[key]) return VARIETY_VISUALS[key]
  // Fallback bila nama tidak dikenal: bentuk berdasar hash, warna hijau default
  const base = key || 'unknown'
  const shapes = ['circle', 'square', 'diamond', 'triangle']
  const sum = base.split('').reduce((a, c) => a + c.charCodeAt(0), 0)
  const shape = shapes[sum % shapes.length]
  return { shape, color: '#2F7E79' }
}

function toneFromConfidence(v) {
  if (typeof v !== 'number' || isNaN(v) || v === 0) return 'blue'
  // Kuning untuk confidence di bawah 70%
  if (v < 0.7) return 'yellow'
  // Hijau untuk confidence 70% ke atas
  return 'green'
}

// Util format selaras dengan halaman Hasil
function formatPercent(num) {
  if (typeof num !== 'number' || isNaN(num)) return '-'
  return `${(num * 100).toFixed(2)}%`
}

function confidenceDisplay(item) {
  if (!item) return '-'
  return item.confidence_percentage || (typeof item.confidence === 'number' ? formatPercent(item.confidence) : '-')
}

// Fallback karakteristik varietas berbasis nama varietas (untuk riwayat lama)
function getLocalVarietyCharacteristics(name) {
  if (!name) return {}
  const n = String(name).trim()
  const map = {
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
    Pilar: { name: 'Pilar', shape: 'bulat telur hingga oval', length_range: '35-55 mm', width_range: '20-40 mm' },
  }
  return map[n] || {}
}

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

function confidenceLevel(item) {
  const v = confidenceValue(item)
  if (v >= 0.95) return 'Sangat yakin'
  if (v >= 0.90) return 'Cukup yakin'
  return 'Tidak pasti'
}

function summarizeModel(model_version) {
  try {
    const obj = typeof model_version === 'string' ? JSON.parse(model_version) : model_version
    if (!obj || typeof obj !== 'object') return '-'
    if (obj.pipeline) return obj.pipeline
    if (obj.artifacts && typeof obj.artifacts === 'object') {
      const names = Object.keys(obj.artifacts)
      return names.length ? names.join('+') : '-'
    }
    return '-'
  } catch {
    return '-'
  }
}

export default function History() {
  const [items, setItems] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [selected, setSelected] = useState(null)
  const [showModal, setShowModal] = useState(false)
  const [isAdmin, setIsAdmin] = useState(false)
  const [profilesMap, setProfilesMap] = useState({})
  const [currentUserId, setCurrentUserId] = useState('')
  const [busyId, setBusyId] = useState('')
  // Filter tanggal (tahun/bulan/hari)
  const [filterYear, setFilterYear] = useState('')
  const [filterMonth, setFilterMonth] = useState('')
  const [filterDay, setFilterDay] = useState('')
  // Filter varietas
  const [filterVariety, setFilterVariety] = useState('')
  const [showDatePicker, setShowDatePicker] = useState(false)
  const [viewMonth, setViewMonth] = useState(new Date().getMonth() + 1)
  const [viewYear, setViewYear] = useState(new Date().getFullYear())
  // Rentang tanggal yang sudah diterapkan (digunakan untuk filter)
  const [dateFrom, setDateFrom] = useState(null)
  const [dateTo, setDateTo] = useState(null)
  // Rentang tanggal yang sedang dipilih dalam kalender (belum diterapkan)
  const [selFrom, setSelFrom] = useState(null)
  const [selTo, setSelTo] = useState(null)
  const [showYearSelect, setShowYearSelect] = useState(false)

  useEffect(() => {
    const load = async () => {
      setError('')
      setLoading(true)
      try {
        if (!supabase) {
          setItems([])
        } else {
          const { data, error } = await supabase
            .from('predictions')
            .select('*')
            .order('created_at', { ascending: false })
          if (error) throw error
          setItems(data || [])
          // Deteksi apakah pengguna sekarang admin, lalu muat nama pengguna untuk semua user_id yang muncul
          try {
            const { data: u } = await supabase.auth.getUser()
            const id = u?.user?.id
            if (id) {
              setCurrentUserId(id)
              const { data: prof } = await supabase.from('profiles').select('role').eq('id', id).single()
              const admin = String(prof?.role || '').trim().toLowerCase() === 'admin'
              setIsAdmin(admin)
              if (admin) {
                const ids = Array.from(new Set((data || []).map(it => it.user_id).filter(Boolean)))
                if (ids.length) {
                  const { data: profs } = await supabase
                    .from('profiles')
                    .select('id, full_name, role')
                    .in('id', ids)
                  const map = {}
                  for (const p of (profs || [])) {
                    map[p.id] = { full_name: p.full_name || '', role: p.role || 'user' }
                  }
                  setProfilesMap(map)
                }
              }
            }
          } catch (_) {}
        }
      } catch (e) {
        setError(e.message || 'Gagal memuat riwayat')
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  const deleteItem = async (it) => {
    if (!it || !it.id) return
    const allowed = !isAdmin && currentUserId && it.user_id === currentUserId
    if (!allowed) return
    if (!window.confirm('Hapus riwayat ini?')) return
    setBusyId(it.id)
    setError('')
    try {
      const { error } = await supabase
        .from('predictions')
        .delete()
        .eq('id', it.id)
        .eq('user_id', currentUserId)
      if (error) throw error
      setItems(prev => prev.filter(x => x.id !== it.id))
    } catch (e) {
      setError(e?.message || 'Gagal menghapus riwayat')
    } finally {
      setBusyId('')
    }
  }

  // Saat modal terbuka, kunci scroll di html & body agar overlay menutup penuh
  useEffect(() => {
    try {
      if (showModal) {
        // Hitung lebar scrollbar sebelum disembunyikan agar layout tidak bergeser
        const sbw = window.innerWidth - document.documentElement.clientWidth
        if (sbw > 0) {
          document.documentElement.style.paddingRight = `${sbw}px`
        }
        document.documentElement.style.overflow = 'hidden'
        document.body.style.overflow = 'hidden'
        document.documentElement.classList.add('modal-open')
        document.body.classList.add('modal-open')
      } else {
        document.documentElement.style.overflow = ''
        document.body.style.overflow = ''
        document.documentElement.style.paddingRight = ''
        document.documentElement.classList.remove('modal-open')
        document.body.classList.remove('modal-open')
      }
    } catch {}
    return () => {
      try {
        document.documentElement.style.overflow = ''
        document.body.style.overflow = ''
        document.documentElement.style.paddingRight = ''
        document.documentElement.classList.remove('modal-open')
        document.body.classList.remove('modal-open')
      } catch {}
    }
  }, [showModal])

  // Reset hierarki pilihan saat induk berubah
  useEffect(() => { setFilterMonth(''); setFilterDay('') }, [filterYear])
  useEffect(() => { setFilterDay('') }, [filterMonth])

  // Opsi dropdown berdasarkan data dan pilihan sebelumnya
  // Gabungkan daftar varietas dari mapping tetap dan dari data agar lengkap
  const staticVarieties = Object.keys(VARIETY_VISUALS)
  const dataVarieties = Array.from(new Set(items.map((it) => String(it.predicted_class || '').trim()).filter(Boolean)))
  const varietyOptions = Array.from(new Set([ ...staticVarieties, ...dataVarieties ])).sort((a, b) => a.localeCompare(b))
  const yearOptions = Array.from(new Set(items.map((it) => {
    const d = new Date(it.created_at)
    return d.getFullYear()
  }))).sort((a, b) => b - a)

  // Bulan selalu dari 1..12 agar pengguna bisa memilih rentang penuh
  const monthOptions = Array.from({ length: 12 }, (_, i) => i + 1)
  const MONTH_NAMES_ID = ['Januari','Februari','Maret','April','Mei','Juni','Juli','Agustus','September','Oktober','November','Desember']

  // Hari mengikuti jumlah hari pada bulan terpilih; jika bulan belum dipilih, tampilkan 1..31
  const daysInMonth = (year, month) => {
    const y = Number(year)
    const m = Number(month)
    if (!y || !m) return 31
    return new Date(y, m, 0).getDate()
  }
  const baseYear = filterYear ? Number(filterYear) : new Date().getFullYear()
  const totalDays = filterMonth ? daysInMonth(baseYear, Number(filterMonth)) : 31
  const dayOptions = Array.from({ length: totalDays }, (_, i) => i + 1)

  // Opsi tahun untuk kalender (gabungan dari data + rentang sekitar tahun berjalan)
  const currentYear = new Date().getFullYear()
  const aroundYears = Array.from({ length: 8 }, (_, i) => currentYear - 6 + i) // kira-kira currentYear-6 .. currentYear+1
  const calYearOptions = Array.from(new Set([ ...yearOptions, ...aroundYears ])).sort((a, b) => a - b)

  // Hasil terfilter
  const startOfDay = (d) => new Date(d.getFullYear(), d.getMonth(), d.getDate(), 0, 0, 0, 0)
  const endOfDay = (d) => new Date(d.getFullYear(), d.getMonth(), d.getDate(), 23, 59, 59, 999)

  const filteredItems = items.filter((it) => {
    const d = new Date(it.created_at)
    const v = String(it.predicted_class || '').trim()
    const yOk = !filterYear || d.getFullYear() === Number(filterYear)
    const mOk = !filterMonth || (d.getMonth() + 1) === Number(filterMonth)
    const dOk = !filterDay || d.getDate() === Number(filterDay)
    const vOk = !filterVariety || v === filterVariety
    let rangeOk = true
    if (dateFrom && dateTo) {
      const t = d.getTime()
      rangeOk = t >= startOfDay(dateFrom).getTime() && t <= endOfDay(dateTo).getTime()
    } else if (dateFrom && !dateTo) {
      const t = d.getTime()
      rangeOk = t >= startOfDay(dateFrom).getTime() && t <= endOfDay(dateFrom).getTime()
    }
    return yOk && mOk && dOk && vOk && rangeOk
  })

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
  // Kunci scroll halaman saat modal terbuka agar latar belakang tetap diam
  useEffect(() => {
    if (!showModal) return
    const originalOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.body.style.overflow = originalOverflow
    }
  }, [showModal])

  return (
    <>
    <section className="card">
      <h1 className="card-title">Riwayat Prediksi</h1>
      <p className="muted">Daftar hasil prediksi yang Anda lakukan.</p>

      {loading && <div className="muted">Memuat…</div>}
      {error && <div className="error" role="alert">{error}</div>}

      {!loading && !error && items.length === 0 && (
        <div className="result">
          <pre>{`Belum ada riwayat.
Setelah melakukan beberapa prediksi, daftar akan muncul di sini.`}</pre>
        </div>
      )}

      {!loading && !error && items.length > 0 && (
        <>
        <div className="filters">
          <button className="filter-button" aria-label="Buka filter tanggal" onClick={() => { setSelFrom(dateFrom); setSelTo(dateTo); setShowDatePicker(v => !v) }}>Filter Tanggal</button>
          {/* Dropdown filter varietas di tengah */}
          <select
            className="filter-select variety-select"
            value={filterVariety}
            onChange={(e) => setFilterVariety(e.target.value)}
            aria-label="Filter varietas"
            style={{ '--variety-ch': String(filterVariety || 'Semua Varietas').length }}
          >
            <option value="">Semua Varietas</option>
            {varietyOptions.map((v) => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
          {/* Tombol reset ditempatkan di samping Varietas */}
          <button className="filter-button" aria-label="Reset filter" onClick={() => { setFilterVariety(''); setFilterYear(''); setFilterMonth(''); setFilterDay(''); clearRange() }}>Reset</button>
          {showDatePicker && (
            <>
              <div className="filter-backdrop" onClick={() => setShowDatePicker(false)} aria-hidden="true" />
              <div className="filter-popover" role="dialog" aria-label="Filter tanggal">
                <div className="calendar" onClick={(e) => { if (showYearSelect && !(e.target.closest && e.target.closest('.year-select'))) setShowYearSelect(false) }}>
                <div className="calendar-header">
                  <button className="nav" onClick={() => { setShowYearSelect(false); goPrevMonth() }} aria-label="Bulan sebelumnya">
                    <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M15 18l-6-6 6-6" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
                  </button>
                  <div className="month-title">
                    {!showYearSelect ? (
                      <button type="button" className="title-button" onClick={() => setShowYearSelect(true)}>
                        {MONTH_NAMES_ID[viewMonth - 1]} {viewYear}
                      </button>
                    ) : (
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
                    )}
                  </div>
                  <button className="nav" onClick={() => { setShowYearSelect(false); goNextMonth() }} aria-label="Bulan berikutnya">
                    <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 6l6 6-6 6" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
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
              {/* Tutup container popover */}
              </div>
            </>
          )}
        </div>

        {filteredItems.length === 0 && (
          <div className="result">
            <pre>{`Tidak ada riwayat yang cocok dengan filter.
Silakan sesuaikan pilihan tahun/bulan/hari atau reset filter.`}</pre>
          </div>
        )}

        {filteredItems.length > 0 && (
        <div className="history-list" role="list">
          {filteredItems.map((it) => {
            // Ikon/border per-varietas
            const vv = varietyVisual(it.predicted_class || '')
            const confVal = confidenceValue(it)
            const confPct = confidenceDisplay(it)
            const badgeClass = confVal >= 0.7 ? 'badge success' : 'badge warning'
            const tone = toneFromConfidence(confVal)
            const toneColor = TONE_COLOR[tone]
            const level = confidenceLevel(it)
            const modelText = summarizeModel(it.model_version)
            const m = it.morphology_info || {}
            return (
              <div
                key={it.id}
                className={`history-item tone-${tone}`}
                role="listitem"
                style={{ borderLeft: `4px solid ${toneColor}`, cursor: 'pointer' }}
                onClick={() => { setSelected(it); setShowModal(true) }}
                aria-label={`Lihat detail gambar untuk ${it.filename || 'item'}`}
              >
                <div
                  className="history-icon"
                  aria-label={`ikon varietas ${it.predicted_class || '-'}`}
                  title={it.predicted_class || ''}
                  style={{ color: ICON_COLOR, fontSize: 18, lineHeight: '18px' }}
                >
                  {glyphForVariety(it.predicted_class)}
                </div>
                <div className="history-main">
                  <div className="history-top">
                    <strong className="history-pred">{it.predicted_class || '-'}</strong>
                    <span className={badgeClass} title="Confidence">{confPct}</span>
                    <span className="small muted">({level})</span>
                    {!isAdmin && currentUserId && it.user_id === currentUserId && (
                      <button
                        className={`btn danger outline icon-square sm`}
                        style={{ marginLeft: 'auto' }}
                        disabled={busyId === it.id}
                        onClick={(e) => { e.stopPropagation(); deleteItem(it) }}
                        aria-label="Hapus riwayat ini"
                        title="Hapus"
                      >
                        <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><path d="M3 6h18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M10 11v6" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M14 11v6" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M9 6V4h6v2" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
                      </button>
                    )}
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
                    <span>•</span>
                    <span className="muted">Model: {modelText}</span>
                  </div>
                  {(m && (m.panjang_daun_mm != null || m.lebar_daun_mm != null || m.rasio_bentuk_daun != null || m.keliling_daun_mm != null || m.panjang_tulang_daun_mm != null)) && (
                    <div className="small muted" aria-label="Pengukuran morfologi">
                      Panjang daun {m.panjang_daun_mm ?? '-'}mm · Lebar daun {m.lebar_daun_mm ?? '-'}mm · Keliling {m.keliling_daun_mm ?? '-'}mm · Panjang tulang {m.panjang_tulang_daun_mm ?? '-'}mm · Rasio bentuk {m.rasio_bentuk_daun ?? '-'}
                    </div>
                  )}
                </div>
              </div>
            )
          })}
        </div>
        )}
        </>
      )}

    </section>
    {showModal && (
      <div
        className="modal-overlay"
        role="dialog"
        aria-modal="true"
        aria-label="Pratinjau gambar deteksi"
        onClick={() => setShowModal(false)}
        style={{
          position: 'fixed', inset: 0, backgroundColor: 'rgba(17,24,39,0.6)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000,
        }}
      >
        <div
          className="modal-content"
          onClick={(e) => e.stopPropagation()}
          style={{ background: '#fff', padding: 16, borderRadius: 8, maxWidth: 800, width: '90%', boxShadow: '0 20px 40px rgba(0,0,0,0.35)' }}
        >
          <div className="modal-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h2 className="card-title" style={{ margin: 0 }}>Detail Riwayat</h2>
            <button className="btn outline" onClick={() => setShowModal(false)}>Tutup</button>
          </div>
          {/* Tidak menampilkan gambar untuk menghindari kebutuhan penyimpanan */}
            <div className="modal-details" style={{ marginTop: 12 }}>
              <div className="summary-top" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <div className="muted">Varietas</div>
                  <div className="summary-name" style={{ fontWeight: 600 }}>{selected?.predicted_class || '-'}</div>
                </div>
                {(() => {
                  const confVal = confidenceValue(selected)
                  const confPct = confidenceDisplay(selected)
                  const badgeClass = confVal >= 0.7 ? 'badge success' : 'badge warning'
                  return (
                    <div className="summary-confidence" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <span className="muted">Confidence</span>
                      <span className={badgeClass} title="Confidence">{confPct}</span>
                    </div>
                  )
                })()}
              </div>
              {(() => {
                const m = selected?.morphology_info || {}
                const vcRaw = selected?.variety_characteristics || {}
                const vcFallback = (!vcRaw || (typeof vcRaw === 'object' && Object.keys(vcRaw).length === 0))
                  ? getLocalVarietyCharacteristics(selected?.predicted_class)
                  : vcRaw
                const vc = vcFallback || {}
                return (
                  <div className="details-grid" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginTop: 8 }}>
                    <div className="card" style={{ padding: 8 }}>
                      <div className="muted" style={{ marginBottom: 6 }}>Morfologi</div>
                      {(() => {
                        const fmt = (v, unit) => (v != null ? `${v}${unit ? ` ${unit}` : ''}` : '-')
                        return (
                          <div className="small" style={{ display: 'grid', gap: 6 }}>
                            <div style={{ display: 'grid', gridTemplateColumns: '140px 1fr', gap: 8 }}>
                              <span style={{ color: 'var(--muted)', fontWeight: 400 }}>Panjang daun</span>
                              <span style={{ color: 'var(--text)', fontWeight: 400 }}>{fmt(m.panjang_daun_mm, 'mm')}</span>
                            </div>
                            <div style={{ display: 'grid', gridTemplateColumns: '140px 1fr', gap: 8 }}>
                              <span style={{ color: 'var(--muted)', fontWeight: 400 }}>Lebar daun</span>
                              <span style={{ color: 'var(--text)', fontWeight: 400 }}>{fmt(m.lebar_daun_mm, 'mm')}</span>
                            </div>
                            <div style={{ display: 'grid', gridTemplateColumns: '140px 1fr', gap: 8 }}>
                              <span style={{ color: 'var(--muted)', fontWeight: 400 }}>Keliling</span>
                              <span style={{ color: 'var(--text)', fontWeight: 400 }}>{fmt(m.keliling_daun_mm, 'mm')}</span>
                            </div>
                            <div style={{ display: 'grid', gridTemplateColumns: '140px 1fr', gap: 8 }}>
                              <span style={{ color: 'var(--muted)', fontWeight: 400 }}>Panjang tulang</span>
                              <span style={{ color: 'var(--text)', fontWeight: 400 }}>{fmt(m.panjang_tulang_daun_mm, 'mm')}</span>
                            </div>
                            <div style={{ display: 'grid', gridTemplateColumns: '140px 1fr', gap: 8 }}>
                              <span style={{ color: 'var(--muted)', fontWeight: 400 }}>Rasio bentuk</span>
                              <span style={{ color: 'var(--text)', fontWeight: 400 }}>{fmt(m.rasio_bentuk_daun, '')}</span>
                            </div>
                          </div>
                        )
                      })()}
                    </div>
                    <div className="card" style={{ padding: 8 }}>
                      <div className="muted" style={{ marginBottom: 6 }}>Karakteristik Varietas</div>
                      {vc && Object.keys(vc).length > 0 ? (
                        <div className="small" style={{ display: 'grid', gap: 6 }}>
                          {Object.entries(vc).map(([k, val]) => (
                            <div key={k} style={{ display: 'grid', gridTemplateColumns: '140px 1fr', gap: 8 }}>
                              <span style={{ color: 'var(--muted)', fontWeight: 400 }}>{k.replace(/_/g,' ')}</span>
                              <span style={{ color: 'var(--text)', fontWeight: 400 }}>{String(val)}</span>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="small muted">Tidak ada karakteristik tersedia.</div>
                      )}
                    </div>
                  </div>
                )
              })()}
          </div>
          <div className="modal-footer" style={{ marginTop: 12 }}>
            <div className="small muted">{selected?.filename || '-'}</div>
            <div className="small muted">Waktu: {selected ? formatDate(selected.created_at) : '-'}</div>
          </div>
        </div>
      </div>
    )}
    </>
  )
}