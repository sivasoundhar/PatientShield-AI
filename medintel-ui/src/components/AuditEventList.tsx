import { useMemo, useState } from "react"
import { Download } from "lucide-react"

import { formatTimestamp, auditStatusBadgeVariant } from "@/lib/format"
import type { AuditEvent } from "@/types"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

const ALL = "__all__"

/** Shared, filterable audit-event table — used by AuditTrail.tsx (standalone page) and Workspace.tsx's Audit tab. */
export function AuditEventList({ events, documentId }: { events: AuditEvent[]; documentId: string }) {
  const [agentFilter, setAgentFilter] = useState<string>(ALL)
  const [actionFilter, setActionFilter] = useState<string>(ALL)

  const agents = useMemo(() => Array.from(new Set(events.map((e) => e.agent_name))).sort(), [events])
  const actions = useMemo(() => Array.from(new Set(events.map((e) => e.action))).sort(), [events])

  const filtered = events.filter(
    (e) => (agentFilter === ALL || e.agent_name === agentFilter) && (actionFilter === ALL || e.action === actionFilter),
  )

  function exportJson() {
    downloadFile(
      `audit-trail-${documentId}.json`,
      JSON.stringify({ document_id: documentId, audit_trail: filtered }, null, 2),
      "application/json",
    )
  }

  function exportCsv() {
    const header = "timestamp,agent_name,action,status,details\n"
    const rows = filtered
      .map((e) =>
        [e.timestamp, e.agent_name, e.action, e.status, JSON.stringify(e.details ?? {})]
          .map((v) => `"${String(v).replace(/"/g, '""')}"`)
          .join(","),
      )
      .join("\n")
    downloadFile(`audit-trail-${documentId}.csv`, header + rows, "text/csv")
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <Select value={agentFilter} onValueChange={setAgentFilter}>
          <SelectTrigger className="w-44">
            <SelectValue placeholder="All agents" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value={ALL}>All agents</SelectItem>
            {agents.map((a) => (
              <SelectItem key={a} value={a}>
                {a}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Select value={actionFilter} onValueChange={setActionFilter}>
          <SelectTrigger className="w-44">
            <SelectValue placeholder="All actions" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value={ALL}>All actions</SelectItem>
            {actions.map((a) => (
              <SelectItem key={a} value={a}>
                {a}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <div className="ml-auto flex gap-2">
          <Button variant="outline" size="sm" onClick={exportJson} disabled={filtered.length === 0}>
            <Download className="size-3.5" /> JSON
          </Button>
          <Button variant="outline" size="sm" onClick={exportCsv} disabled={filtered.length === 0}>
            <Download className="size-3.5" /> CSV
          </Button>
        </div>
      </div>

      {filtered.length === 0 ? (
        <p className="py-10 text-center text-sm text-muted-foreground">No audit events match these filters.</p>
      ) : (
        <ul className="divide-y divide-border rounded-lg border border-border">
          {filtered.map((event, i) => (
            <li key={i} className="flex items-start gap-3 px-4 py-3 text-sm">
              <Badge variant={auditStatusBadgeVariant(event.status)}>{event.status}</Badge>
              <div className="min-w-0 flex-1">
                <p className="font-medium text-foreground">
                  {event.agent_name} <span className="text-muted-foreground">— {event.action}</span>
                </p>
                {event.details && Object.keys(event.details).length > 0 && (
                  <p className="mt-0.5 truncate text-xs text-muted-foreground">
                    {Object.entries(event.details)
                      .map(([k, v]) => `${k}: ${v}`)
                      .join(" · ")}
                  </p>
                )}
              </div>
              <span className="shrink-0 whitespace-nowrap text-xs text-muted-foreground">
                {formatTimestamp(event.timestamp)}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

function downloadFile(filename: string, content: string, mimeType: string) {
  const blob = new Blob([content], { type: mimeType })
  const url = URL.createObjectURL(blob)
  const a = window.document.createElement("a")
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}
