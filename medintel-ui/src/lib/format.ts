import type { AuditStatus, ProcessingStatus } from "@/types"

/** Badge variant for a document's processing_status — kept in one place so every page renders status identically. */
export function statusBadgeVariant(status: ProcessingStatus): "success" | "warning" | "danger" | "secondary" {
  switch (status) {
    case "completed":
      return "success"
    case "processing":
      return "warning"
    case "failed":
      return "danger"
    case "uploaded":
      return "secondary"
  }
}

/** Badge variant for an AuditEvent's status field. */
export function auditStatusBadgeVariant(status: AuditStatus): "success" | "danger" | "secondary" {
  switch (status) {
    case "success":
      return "success"
    case "error":
      return "danger"
    case "skipped":
      return "secondary"
  }
}

/** Human-readable relative-ish timestamp for activity feeds (e.g. "2 min ago"), falling back to a locale time for anything over a day old. */
export function formatRelativeTime(iso: string): string {
  const date = new Date(iso)
  const diffMs = Date.now() - date.getTime()
  const diffMin = Math.round(diffMs / 60_000)

  if (diffMin < 1) return "just now"
  if (diffMin < 60) return `${diffMin} min ago`
  const diffHours = Math.round(diffMin / 60)
  if (diffHours < 24) return `${diffHours}h ago`
  return date.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" })
}

/** Absolute timestamp for audit-trail rows, where precision matters more than readability. */
export function formatTimestamp(iso: string): string {
  return new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "medium" })
}

/** 0.0-1.0 confidence/priority score as a percentage string, e.g. "87%". */
export function formatScore(score: number): string {
  return `${Math.round(score * 100)}%`
}

/** Bytes to a human-readable size (matches the "Max 50MB" style copy used on Dashboard). */
export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}
