import { useEffect, useRef, useState } from "react"
import { useParams, useNavigate, Link } from "react-router-dom"
import { CheckCircle2, CircleDashed, XCircle, Loader2, FileWarning } from "lucide-react"

import { api, getErrorMessage } from "@/services/api"
import { formatTimestamp } from "@/lib/format"
import { cacheProcessResult } from "@/lib/processResultsCache"
import type { AuditEvent, DocumentResponse, ProcessingStatus } from "@/types"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"

/**
 * The real 5-agent pipeline (src/orchestrator/pipeline.py) — deliberately
 * NOT the 7-stage version shown in the UI_ref mockups, which included a
 * "De-identification Agent" and "Reviewer Agent" that don't exist in the
 * backend. Showing agents that aren't real would make this display
 * decorative rather than a genuine reflection of what ran.
 */
const AGENT_STAGES = [
  { key: "planner", label: "Planner Agent", description: "Orchestrating the workflow" },
  { key: "phi_agent", label: "PHI Reasoner Agent", description: "Detecting protected health information" },
  { key: "clinical_agent", label: "Clinical Agent", description: "Extracting diagnoses, medications, findings" },
  { key: "knowledge_agent", label: "Knowledge Agent", description: "Indexing for Q&A retrieval" },
  { key: "audit_agent", label: "Audit Agent", description: "Compiling the compliance record" },
] as const

const POLL_INTERVAL_MS = 1000

function StageIcon({ event }: { event: AuditEvent | undefined }) {
  if (!event) return <CircleDashed className="size-5 text-muted-foreground/50" />
  if (event.status === "success") return <CheckCircle2 className="size-5 text-status-success" />
  if (event.status === "error") return <XCircle className="size-5 text-status-danger" />
  return <CircleDashed className="size-5 text-status-warning" /> // "skipped" — Day 4-7 placeholder, not yet implemented
}

export function Processing() {
  const { documentId } = useParams<{ documentId: string }>()
  const navigate = useNavigate()

  const [document, setDocument] = useState<DocumentResponse | null>(null)
  const [events, setEvents] = useState<AuditEvent[]>([])
  const [pipelineStatus, setPipelineStatus] = useState<ProcessingStatus | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isBusy, setIsBusy] = useState(false)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current)
    }
  }, [])

  useEffect(() => {
    if (!documentId) return
    let cancelled = false

    async function loadInitialState() {
      try {
        const doc = await api.getDocument(documentId!)
        if (cancelled) return
        setDocument(doc)
        setPipelineStatus(doc.processing_status)

        if (doc.processing_status === "processing") {
          // Someone else (or a previous page load) already kicked this off —
          // poll rather than re-trigger a second /process run.
          startPolling(documentId!)
        } else if (doc.processing_status === "completed" || doc.processing_status === "failed") {
          const trail = await api.getAuditTrail(documentId!)
          if (!cancelled) setEvents(trail.audit_trail)
        }
      } catch (err) {
        if (!cancelled) setError(getErrorMessage(err))
      }
    }

    loadInitialState()
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [documentId])

  function startPolling(id: string) {
    if (pollRef.current) clearInterval(pollRef.current)
    pollRef.current = setInterval(async () => {
      try {
        const status = await api.getPipelineStatus(id)
        setEvents(status.events)
        setPipelineStatus(status.status)
        if (status.status !== "processing" && pollRef.current) {
          clearInterval(pollRef.current)
          pollRef.current = null
        }
      } catch (err) {
        setError(getErrorMessage(err))
        if (pollRef.current) clearInterval(pollRef.current)
      }
    }, POLL_INTERVAL_MS)
  }

  async function runPipeline() {
    if (!documentId) return
    setIsBusy(true)
    setError(null)
    setPipelineStatus("processing")
    try {
      const result = await api.processDocument(documentId)
      cacheProcessResult(documentId, result)
      setEvents(result.audit_events)
      setPipelineStatus(result.processing_status)
      const doc = await api.getDocument(documentId)
      setDocument(doc)
    } catch (err) {
      setError(getErrorMessage(err))
      setPipelineStatus("failed")
    } finally {
      setIsBusy(false)
    }
  }

  if (!documentId) {
    return (
      <EmptyState message="Select a document from the Dashboard to view its processing pipeline." />
    )
  }

  const eventByAgent = new Map(events.map((e) => [e.agent_name, e]))
  const completedCount = AGENT_STAGES.filter((s) => eventByAgent.has(s.key)).length
  const progressPct = (completedCount / AGENT_STAGES.length) * 100
  const isRunning = pipelineStatus === "processing"

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs text-muted-foreground">
            <Link to="/" className="hover:underline">
              Documents
            </Link>{" "}
            / Processing
          </p>
          <h1 className="mt-1 text-2xl font-semibold text-foreground">
            {document?.filename ?? "Processing Document"}
          </h1>
        </div>
        {pipelineStatus && (
          <Badge
            variant={pipelineStatus === "completed" ? "success" : pipelineStatus === "failed" ? "danger" : "warning"}
          >
            {pipelineStatus}
          </Badge>
        )}
      </div>

      {error && (
        <div className="flex items-center gap-2 rounded-md bg-status-danger-bg px-4 py-3 text-sm text-status-danger">
          <FileWarning className="size-4 shrink-0" />
          {error}
        </div>
      )}

      <Card>
        <CardHeader>
          <CardTitle>AI Processing Pipeline</CardTitle>
          {pipelineStatus === "uploaded" && (
            <Button onClick={runPipeline} disabled={isBusy} size="sm">
              {isBusy ? (
                <>
                  <Loader2 className="size-4 animate-spin" /> Starting…
                </>
              ) : (
                "Start Processing"
              )}
            </Button>
          )}
        </CardHeader>
        <CardContent>
          <ul className="flex flex-col gap-4">
            {AGENT_STAGES.map((stage) => {
              const event = eventByAgent.get(stage.key)
              return (
                <li key={stage.key} className="flex items-center gap-3">
                  <StageIcon event={event} />
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-medium text-foreground">{stage.label}</p>
                    <p className="truncate text-xs text-muted-foreground">
                      {(event?.details?.note as string | undefined) ?? stage.description}
                    </p>
                  </div>
                  {isRunning && !event && <Loader2 className="size-4 shrink-0 animate-spin text-muted-foreground" />}
                </li>
              )
            })}
          </ul>

          <div className="mt-6">
            <div className="mb-1.5 flex items-center justify-between text-xs text-muted-foreground">
              <span>Overall Progress</span>
              <span>{Math.round(progressPct)}%</span>
            </div>
            <Progress value={progressPct} />
          </div>
        </CardContent>
      </Card>

      {events.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Event Log</CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            <ul className="divide-y divide-border">
              {events.map((event, i) => (
                <li key={i} className="flex items-start gap-3 px-5 py-3 text-sm">
                  <StageIcon event={event} />
                  <div className="min-w-0 flex-1">
                    <p className="font-medium text-foreground">
                      {event.agent_name} — {event.action}
                    </p>
                    {event.details?.note != null && (
                      <p className="text-xs text-muted-foreground">{String(event.details.note)}</p>
                    )}
                  </div>
                  <span className="shrink-0 text-xs text-muted-foreground">{formatTimestamp(event.timestamp)}</span>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}

      {pipelineStatus === "completed" && (
        <div className="flex justify-end">
          <Button onClick={() => navigate(`/workspace/${documentId}`)}>View Results</Button>
        </div>
      )}
    </div>
  )
}

function EmptyState({ message }: { message: string }) {
  return (
    <div className="flex flex-col items-center gap-3 rounded-xl border border-dashed border-border py-24 text-center">
      <FileWarning className="size-8 text-muted-foreground/60" />
      <p className="text-sm text-muted-foreground">{message}</p>
      <Button asChild variant="outline" size="sm">
        <Link to="/">Go to Dashboard</Link>
      </Button>
    </div>
  )
}
