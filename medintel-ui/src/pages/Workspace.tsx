import { useEffect, useState } from "react"
import { useParams, Link, useNavigate } from "react-router-dom"
import { FileWarning, ShieldAlert } from "lucide-react"

import { api, getErrorMessage } from "@/services/api"
import { getCachedProcessResult } from "@/lib/processResultsCache"
import { statusBadgeVariant, formatScore } from "@/lib/format"
import type { DocumentResponse, ProcessResponse } from "@/types"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ChatPanel } from "@/components/ChatPanel"
import { AuditEventList } from "@/components/AuditEventList"

export function Workspace() {
  const { documentId } = useParams<{ documentId: string }>()
  const navigate = useNavigate()

  const [document, setDocument] = useState<DocumentResponse | null>(null)
  const [cachedResult, setCachedResult] = useState<ProcessResponse | null>(null)
  const [auditEvents, setAuditEvents] = useState<Awaited<ReturnType<typeof api.getAuditTrail>>["audit_trail"]>([])
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!documentId) return
    let cancelled = false

    async function load() {
      try {
        const [doc, trail] = await Promise.all([api.getDocument(documentId!), api.getAuditTrail(documentId!)])
        if (cancelled) return
        setDocument(doc)
        setAuditEvents(trail.audit_trail)
        setCachedResult(getCachedProcessResult(documentId!))
      } catch (err) {
        if (!cancelled) setError(getErrorMessage(err))
      }
    }
    load()
    return () => {
      cancelled = true
    }
  }, [documentId])

  if (!documentId) {
    return (
      <div className="flex flex-col items-center gap-3 rounded-xl border border-dashed border-border py-24 text-center">
        <FileWarning className="size-8 text-muted-foreground/60" />
        <p className="text-sm text-muted-foreground">Select a document from the Dashboard to open its workspace.</p>
        <Button asChild variant="outline" size="sm">
          <Link to="/">Go to Dashboard</Link>
        </Button>
      </div>
    )
  }

  if (error) {
    return <p className="text-sm text-status-danger">{error}</p>
  }

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs text-muted-foreground">
            <Link to="/" className="hover:underline">
              Documents
            </Link>{" "}
            / {document?.filename ?? "…"}
          </p>
          <h1 className="mt-1 text-2xl font-semibold text-foreground">{document?.filename ?? "Loading…"}</h1>
        </div>
        {document && (
          <div className="flex items-center gap-2">
            <Badge variant={statusBadgeVariant(document.processing_status)}>{document.processing_status}</Badge>
            <Button variant="outline" size="sm" onClick={() => navigate(`/processing/${documentId}`)}>
              View Pipeline
            </Button>
          </div>
        )}
      </div>

      <Tabs defaultValue="summary">
        <TabsList>
          <TabsTrigger value="summary">Summary</TabsTrigger>
          <TabsTrigger value="phi">PHI Detection</TabsTrigger>
          <TabsTrigger value="chat">Chat</TabsTrigger>
          <TabsTrigger value="audit">Audit</TabsTrigger>
        </TabsList>

        <TabsContent value="summary">
          <Card>
            <CardContent className="p-5">
              {cachedResult?.clinical_analysis ? (
                <div className="flex flex-col gap-4">
                  <p className="text-sm text-foreground">{cachedResult.clinical_analysis.summary}</p>
                  <ul className="flex flex-col gap-2">
                    {cachedResult.clinical_analysis.findings.map((f, i) => (
                      <li key={i} className="flex items-center justify-between rounded-md border border-border px-3 py-2 text-sm">
                        <span>
                          <span className="font-medium text-foreground">{f.category}</span>{" "}
                          <span className="text-muted-foreground">— {f.value}</span>
                        </span>
                        <span className="text-xs text-muted-foreground">{formatScore(f.priority_score)} priority</span>
                      </li>
                    ))}
                  </ul>
                </div>
              ) : (
                <NotYetAvailable
                  note="Clinical Agent lands Day 5 — no clinical findings to show yet."
                  documentId={documentId}
                  navigate={navigate}
                />
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="phi">
          <Card>
            <CardContent className="p-5">
              {cachedResult ? (
                <div className="flex flex-col gap-4">
                  {cachedResult.phi_detected && cachedResult.phi_detected.entities.length > 0 ? (
                    <div className="flex items-start gap-2 rounded-md bg-status-success-bg px-3 py-2 text-sm text-status-success">
                      <ShieldAlert className="size-4 shrink-0" />
                      {cachedResult.phi_detected.total_count} PHI {cachedResult.phi_detected.total_count === 1 ? "entity" : "entities"} detected and masked below.
                    </div>
                  ) : (
                    <div className="flex items-start gap-2 rounded-md bg-status-info-bg px-3 py-2 text-sm text-status-info">
                      <ShieldAlert className="size-4 shrink-0" />
                      No PHI entities detected in this document (or none met the confidence threshold).
                    </div>
                  )}

                  <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                    <div>
                      <p className="mb-1.5 text-xs font-medium text-muted-foreground">Original Text</p>
                      <pre className="max-h-96 overflow-auto whitespace-pre-wrap rounded-md border border-border bg-muted/40 p-3 text-xs text-foreground">
                        {cachedResult.original_text}
                      </pre>
                    </div>
                    <div>
                      <p className="mb-1.5 text-xs font-medium text-muted-foreground">De-Identified Text</p>
                      <pre className="max-h-96 overflow-auto whitespace-pre-wrap rounded-md border border-border bg-muted/40 p-3 text-xs text-foreground">
                        {cachedResult.de_identified_text}
                      </pre>
                    </div>
                  </div>

                  {cachedResult.phi_detected && cachedResult.phi_detected.entities.length > 0 && (
                    <div>
                      <p className="mb-1.5 text-xs font-medium text-muted-foreground">Detected Entities</p>
                      <ul className="flex flex-col gap-1.5">
                        {cachedResult.phi_detected.entities.map((entity, i) => (
                          <li key={i} className="flex items-start justify-between gap-3 rounded-md border border-border px-3 py-2 text-sm">
                            <div className="min-w-0">
                              <span className="font-medium text-foreground">[{entity.entity_type}]</span>{" "}
                              <span className="text-muted-foreground">{entity.reasoning}</span>
                            </div>
                            <span className="shrink-0 text-xs text-muted-foreground">{formatScore(entity.confidence)}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              ) : (
                <NotYetAvailable
                  note="No PHI detection results cached for this document in this session."
                  documentId={documentId}
                  navigate={navigate}
                />
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="chat">
          <Card className="h-[600px]">
            <ChatPanel documentId={documentId} />
          </Card>
        </TabsContent>

        <TabsContent value="audit">
          <Card>
            <CardContent className="p-5">
              <AuditEventList events={auditEvents} documentId={documentId} />
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

function NotYetAvailable({
  note,
  documentId,
  navigate,
}: {
  note: string
  documentId: string
  navigate: ReturnType<typeof useNavigate>
}) {
  return (
    <div className="flex flex-col items-center gap-3 py-10 text-center">
      <p className="text-sm text-muted-foreground">{note}</p>
      <Button variant="outline" size="sm" onClick={() => navigate(`/processing/${documentId}`)}>
        Go to Processing
      </Button>
    </div>
  )
}
