import { useCallback, useEffect, useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import {
  Search,
  Zap,
  Activity,
  TrendingUp,
  TrendingDown,
  Minus,
  Sparkles,
  Database,
  Layers,
} from "lucide-react"

const API_BASE = import.meta.env.VITE_API_BASE ?? "/api"

interface SearchResult {
  id: string
  title: string
  description: string
  score: number
  original_rank: number
  reranked_rank: number
  rank_change: number
  semantic_score?: number | null
  keyword_score?: number | null
}

type ColumnMode = "lexical" | "neural" | "hybrid"

interface TripleSearchResponse {
  lexical: SearchResult[]
  neural: SearchResult[]
  hybrid: SearchResult[]
  latency_ms: number
  cached?: boolean
  search_fingerprint: string
}

function renderRankChange(change: number) {
  if (change > 0) {
    return (
      <span className="flex items-center text-emerald-400 font-medium">
        <TrendingUp className="w-4 h-4 mr-1" /> {change}
      </span>
    )
  }
  if (change < 0) {
    return (
      <span className="flex items-center text-rose-400 font-medium">
        <TrendingDown className="w-4 h-4 mr-1" /> {Math.abs(change)}
      </span>
    )
  }
  return (
    <span className="flex items-center text-slate-500 font-medium">
      <Minus className="w-4 h-4 mr-1" />
    </span>
  )
}

const COLUMN_META: Record<
  ColumnMode,
  { label: string; icon: typeof Sparkles; accent: string; border: string }
> = {
  lexical: {
    label: "Lexical",
    icon: Database,
    accent: "text-amber-300",
    border: "border-amber-500/30",
  },
  neural: {
    label: "Neural",
    icon: Sparkles,
    accent: "text-blue-300",
    border: "border-blue-500/35",
  },
  hybrid: {
    label: "Hybrid",
    icon: Layers,
    accent: "text-teal-300",
    border: "border-teal-500/30",
  },
}

function ResultCard({
  res,
  idx,
  column,
}: {
  res: SearchResult
  idx: number
  column: ColumnMode
}) {
  const neuralHighlight = column === "neural" && idx < 3
  return (
    <motion.article
      layout
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.98 }}
      transition={{ duration: 0.2, delay: Math.min(idx * 0.03, 0.35) }}
      className={`relative bg-slate-800/35 border ${
        neuralHighlight
          ? "border-blue-500/40 ring-1 ring-blue-500/15"
          : "border-slate-700/50"
      } backdrop-blur-sm p-4 rounded-xl hover:bg-slate-800/55 transition-all group`}
    >
      {neuralHighlight && (
        <div className="absolute -top-2.5 -left-2.5 w-8 h-8 bg-gradient-to-br from-blue-400 to-indigo-600 text-white rounded-full flex items-center justify-center font-bold text-xs shadow-lg z-10">
          #{idx + 1}
        </div>
      )}

      <div className="flex justify-between items-start gap-3 mb-2">
        <h2 className="text-sm font-semibold text-white group-hover:text-blue-300 transition-colors line-clamp-2 leading-snug">
          {res.title}
        </h2>
        <div className="shrink-0 bg-slate-900/90 px-2 py-1 rounded-md text-[10px] font-mono text-blue-300 border border-slate-700/80">
          {res.score.toFixed(4)}
        </div>
      </div>

      <p className="text-slate-400 text-xs leading-relaxed mb-3 line-clamp-3">{res.description}</p>

      <div className="flex flex-col gap-1.5 pt-3 border-t border-slate-700/50 text-[10px]">
        <div className="text-slate-500 flex flex-wrap gap-x-2 gap-y-0.5">
          <span>
            {res.original_rank} → <span className="text-white font-medium">{res.reranked_rank}</span>
          </span>
          {res.semantic_score != null && res.keyword_score != null && (
            <>
              <span className="text-slate-600">·</span>
              <span>
                sem {res.semantic_score.toFixed(3)} · kw {res.keyword_score.toFixed(3)}
              </span>
            </>
          )}
        </div>
        <div>{renderRankChange(res.rank_change)}</div>
      </div>
    </motion.article>
  )
}

export default function App() {
  const [query, setQuery] = useState("")
  const [topK, setTopK] = useState(20)
  const [lexical, setLexical] = useState<SearchResult[]>([])
  const [neural, setNeural] = useState<SearchResult[]>([])
  const [hybrid, setHybrid] = useState<SearchResult[]>([])
  const [latency, setLatency] = useState(0)
  const [cached, setCached] = useState(false)
  const [fingerprint, setFingerprint] = useState("")
  const [error, setError] = useState<string | null>(null)
  const [isSearching, setIsSearching] = useState(false)
  const [neuralExplain, setNeuralExplain] = useState<string | null>(null)
  const [explainError, setExplainError] = useState<string | null>(null)
  const [explainLoading, setExplainLoading] = useState(false)
  const [searchedQuery, setSearchedQuery] = useState("")

  const runSearch = useCallback(async () => {
    const q = query.trim()
    if (!q) return
    setIsSearching(true)
    setError(null)
    setNeuralExplain(null)
    setExplainError(null)
    try {
      const res = await fetch(`${API_BASE}/search/triple`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, top_k: topK }),
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error((err as { detail?: string }).detail || res.statusText)
      }
      const data: TripleSearchResponse = await res.json()
      setLexical(data.lexical ?? [])
      setNeural(data.neural ?? [])
      setHybrid(data.hybrid ?? [])
      setLatency(data.latency_ms ?? 0)
      setCached(!!data.cached)
      setFingerprint(data.search_fingerprint ?? `triple:${q}:${topK}`)
      setSearchedQuery(q)
    } catch (e) {
      setLexical([])
      setNeural([])
      setHybrid([])
      setError(e instanceof Error ? e.message : "Request failed")
    } finally {
      setIsSearching(false)
    }
  }, [query, topK])

  useEffect(() => {
    if (neural.length === 0) {
      setNeuralExplain(null)
      setExplainError(null)
      return
    }
    const top3 = neural.slice(0, 3).map((r) => ({
      title: r.title,
      description: r.description,
      score: r.score,
    }))
    let cancelled = false
    setExplainLoading(true)
    setNeuralExplain(null)
    setExplainError(null)
    fetch(`${API_BASE}/neural-explain`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: searchedQuery, top_three: top3 }),
    })
      .then(async (res) => {
        const data = await res.json()
        if (cancelled) return
        if (data.error) setExplainError(data.error)
        else setNeuralExplain(data.explanation || "")
      })
      .catch((e) => {
        if (!cancelled) setExplainError(e instanceof Error ? e.message : "Explain failed")
      })
      .finally(() => {
        if (!cancelled) setExplainLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [neural, fingerprint, searchedQuery])

  const hasResults = lexical.length > 0 || neural.length > 0 || hybrid.length > 0

  const columns: { mode: ColumnMode; results: SearchResult[] }[] = [
    { mode: "lexical", results: lexical },
    { mode: "neural", results: neural },
    { mode: "hybrid", results: hybrid },
  ]

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950 text-slate-200">
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-blue-950/40 via-transparent to-transparent pointer-events-none" />
      <div className="relative max-w-[1600px] mx-auto px-4 sm:px-6 py-10 space-y-10 font-sans">
        <header className="text-center space-y-3">
          <motion.div
            initial={{ opacity: 0, y: -12 }}
            animate={{ opacity: 1, y: 0 }}
            className="inline-flex items-center justify-center p-3 bg-blue-500/15 rounded-2xl border border-blue-500/20 mb-2"
          >
            <Zap className="w-9 h-9 text-blue-400" />
          </motion.div>
          <h1 className="text-4xl sm:text-5xl font-bold tracking-tight text-white">
            NeuralSearch Prime
          </h1>
          <p className="text-slate-400 text-lg max-w-2xl mx-auto">
            One search — three columns: lexical, neural, and hybrid rankings from a single backend pass.
          </p>
        </header>

        <motion.section
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-slate-800/40 backdrop-blur-xl border border-slate-700/60 rounded-2xl p-6 sm:p-8 shadow-2xl shadow-black/40 space-y-6 max-w-3xl mx-auto"
        >
          <div className="relative">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-500" />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && runSearch()}
              placeholder="e.g. stainless steel water bottle 32oz"
              className="w-full bg-slate-900/70 border border-slate-600/80 text-white rounded-xl pl-12 pr-4 py-3.5 focus:outline-none focus:ring-2 focus:ring-blue-500/80 focus:border-blue-500/50 transition-all placeholder:text-slate-600"
            />
          </div>

          <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-6 border-t border-slate-700/50 pt-6">
            <div className="flex-1 flex items-center gap-4">
              <span className="text-sm font-medium text-slate-400 whitespace-nowrap">
                Top K: {topK}
              </span>
              <input
                type="range"
                min={10}
                max={50}
                value={topK}
                onChange={(e) => setTopK(parseInt(e.target.value, 10))}
                className="w-full accent-blue-500 cursor-pointer h-2"
              />
            </div>
            <motion.button
              type="button"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={runSearch}
              disabled={isSearching}
              className="sm:w-auto bg-blue-600 hover:bg-blue-500 text-white px-10 py-3.5 rounded-xl font-semibold flex items-center justify-center gap-2 transition-all disabled:opacity-60 shadow-lg shadow-blue-900/30"
            >
              {isSearching ? (
                <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              ) : (
                <Search className="w-5 h-5" />
              )}
              {isSearching ? "Searching…" : "Search"}
            </motion.button>
          </div>
        </motion.section>

        {error && (
          <div className="rounded-xl border border-rose-500/40 bg-rose-950/40 text-rose-200 px-4 py-3 text-sm max-w-3xl mx-auto">
            {error}
          </div>
        )}

        {hasResults && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 bg-emerald-500/10 border border-emerald-500/25 text-emerald-100/90 px-5 py-4 rounded-xl max-w-3xl mx-auto"
          >
            <div className="flex items-center gap-2">
              <Activity className="w-5 h-5 shrink-0" />
              <span>
                {neural.length} results per column
                {cached && (
                  <span className="ml-2 text-emerald-400/90 text-xs">(cached)</span>
                )}
              </span>
            </div>
            <div className="text-sm opacity-90">
              Latency: <strong className="text-white">{latency.toFixed(0)} ms</strong>
            </div>
          </motion.div>
        )}

        {neural.length > 0 && (
          <div className="rounded-2xl border border-violet-500/25 bg-violet-950/30 p-5 backdrop-blur-sm max-w-3xl mx-auto">
            <div className="flex items-center gap-2 text-violet-200 font-semibold mb-2">
              <Sparkles className="w-5 h-5" />
              Why these top 3 (neural column)? — Groq
            </div>
            {explainLoading && (
              <p className="text-slate-400 text-sm animate-pulse">Generating explanation…</p>
            )}
            {explainError && <p className="text-amber-300/90 text-sm">{explainError}</p>}
            {!explainLoading && neuralExplain && (
              <div className="text-slate-300 text-sm leading-relaxed whitespace-pre-wrap">
                {neuralExplain}
              </div>
            )}
            {!explainLoading && !neuralExplain && !explainError && (
              <p className="text-slate-500 text-sm">No explanation yet.</p>
            )}
          </div>
        )}

        {hasResults && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 lg:gap-5 items-start">
            {columns.map(({ mode, results }) => {
              const meta = COLUMN_META[mode]
              const Icon = meta.icon
              return (
                <div
                  key={mode}
                  className={`rounded-2xl border ${meta.border} bg-slate-900/40 overflow-hidden flex flex-col min-h-[200px]`}
                >
                  <div
                    className={`sticky top-0 z-20 px-4 py-3 border-b border-slate-700/60 bg-slate-900/95 backdrop-blur flex items-center gap-2 ${meta.accent}`}
                  >
                    <Icon className="w-5 h-5 shrink-0 opacity-90" />
                    <span className="font-bold tracking-tight">{meta.label}</span>
                    <span className="ml-auto text-xs text-slate-500 font-normal">
                      {mode === "lexical" && "keyword overlap"}
                      {mode === "neural" && "0.7·sem + 0.3·kw"}
                      {mode === "hybrid" && "0.5·sem + 0.5·kw"}
                    </span>
                  </div>
                  <div className="p-3 space-y-3 flex-1">
                    <AnimatePresence mode="popLayout">
                      {results.map((res, idx) => (
                        <ResultCard
                          key={`${fingerprint}-${mode}-${res.reranked_rank}-${res.id}`}
                          res={res}
                          idx={idx}
                          column={mode}
                        />
                      ))}
                    </AnimatePresence>
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}
