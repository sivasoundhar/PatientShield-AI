import { useEffect, useState } from "react"
import { useParams, useNavigate } from "react-router-dom"
import { FileText } from "lucide-react"

import { api, getErrorMessage } from "@/services/api"
import { getRecentDocuments } from "@/lib/recentDocuments"
import type { AuditEvent } from "@/types"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { AuditEventList } from "@/components/AuditEventList"

/**
 * Audit is document-scoped in the backend (GET /audit/{document_id} — there
 * is no global "all documents" audit endpoint, same gap as the Dashboard's
 * recent-documents list). This page uses the same client-tracked recent
 * documents list as a picker rather than pretending a global view exists.
 */
export function AuditTrail() {
  const { documentId } = useParams<{ documentId: string }>()
  const navigate = useNavigate()
  const recentDocs = getRecentDocuments()

  const [events, setEvents] = useState<AuditEvent[]>([])
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!documentId) return
    let cancelled = false
    api
      .getAuditTrail(documentId)
      .then((trail) => {
        if (!cancelled) setEvents(trail.audit_trail)
      })
      .catch((err) => {
        if (!cancelled) setError(getErrorMessage(err))
      })
    return () => {
      cancelled = true
    }
  }, [documentId])

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-semibold text-foreground">Audit Trail</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Every agent decision, logged with a timestamp — the HIPAA-facing compliance record.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Document</CardTitle>
        </CardHeader>
        <CardContent>
          {recentDocs.length === 0 ? (
            <p className="text-sm text-muted-foreground">No documents uploaded yet in this browser.</p>
          ) : (
            <Select value={documentId} onValueChange={(id) => navigate(`/audit/${id}`)}>
              <SelectTrigger className="w-80">
                <SelectValue placeholder="Select a document…" />
              </SelectTrigger>
              <SelectContent>
                {recentDocs.map((doc) => (
                  <SelectItem key={doc.documentId} value={doc.documentId}>
                    <FileText className="size-3.5" /> {doc.filename}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
        </CardContent>
      </Card>

      {error && <p className="text-sm text-status-danger">{error}</p>}

      {documentId && (
        <Card>
          <CardContent className="p-5">
            <AuditEventList events={events} documentId={documentId} />
          </CardContent>
        </Card>
      )}
    </div>
  )
}
