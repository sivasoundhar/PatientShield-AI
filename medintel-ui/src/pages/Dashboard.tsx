import { useCallback, useEffect, useRef, useState } from "react"
import { useNavigate } from "react-router-dom"
import { UploadCloud, FileText, Loader2, AlertCircle } from "lucide-react"

import { api, getErrorMessage } from "@/services/api"
import { useCurrentDocument } from "@/context/DocumentContext"
import { getRecentDocuments, trackUploadedDocument, type RecentDocumentEntry } from "@/lib/recentDocuments"
import { statusBadgeVariant, formatRelativeTime } from "@/lib/format"
import type { DocumentResponse } from "@/types"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"

const ACCEPTED_EXTENSIONS = [".pdf", ".docx", ".txt"]

/** One tracked upload, hydrated with its live status from GET /documents/{id}. */
interface RecentRow extends RecentDocumentEntry {
  document?: DocumentResponse
  loadError?: string
}

export function Dashboard() {
  const navigate = useNavigate()
  const { setCurrentDocumentId } = useCurrentDocument()
  const fileInputRef = useRef<HTMLInputElement>(null)

  const [isDragging, setIsDragging] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [rows, setRows] = useState<RecentRow[]>(() => getRecentDocuments())

  // Hydrate each tracked upload with its live status. Runs once on mount and
  // again whenever the tracked list changes (e.g. right after an upload) —
  // this is what turns "documents this browser uploaded" into an accurate
  // recent-documents view instead of stale local state.
  useEffect(() => {
    let cancelled = false
    async function hydrate() {
      const tracked = getRecentDocuments()
      const hydrated = await Promise.all(
        tracked.map(async (entry): Promise<RecentRow> => {
          try {
            const document = await api.getDocument(entry.documentId)
            return { ...entry, document }
          } catch (error) {
            return { ...entry, loadError: getErrorMessage(error) }
          }
        }),
      )
      if (!cancelled) setRows(hydrated)
    }
    hydrate()
    return () => {
      cancelled = true
    }
  }, [])

  const handleFiles = useCallback(
    async (files: FileList | null) => {
      const file = files?.[0]
      if (!file) return

      const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase()
      if (!ACCEPTED_EXTENSIONS.includes(ext)) {
        setUploadError(`Unsupported file type '${ext}'. Supported: ${ACCEPTED_EXTENSIONS.join(", ")}`)
        return
      }

      setIsUploading(true)
      setUploadError(null)
      try {
        const uploaded = await api.uploadDocument(file)
        trackUploadedDocument({
          documentId: uploaded.document_id,
          filename: uploaded.filename,
          uploadedAt: uploaded.timestamp,
        })
        setCurrentDocumentId(uploaded.document_id)
        navigate(`/processing/${uploaded.document_id}`)
      } catch (error) {
        setUploadError(getErrorMessage(error))
      } finally {
        setIsUploading(false)
      }
    },
    [navigate, setCurrentDocumentId],
  )

  function openDocument(documentId: string) {
    setCurrentDocumentId(documentId)
    navigate(`/workspace/${documentId}`)
  }

  const processed = rows.filter((r) => r.document?.processing_status === "completed").length
  const inProgress = rows.filter((r) => r.document?.processing_status === "processing").length
  const failed = rows.filter((r) => r.document?.processing_status === "failed").length

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-semibold text-foreground">Dashboard</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Upload a medical document to detect PHI, extract clinical findings, and enable Q&amp;A.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[1.4fr_1fr]">
        {/* Upload dropzone */}
        <Card>
          <CardContent className="p-0">
            <div
              onDragOver={(e) => {
                e.preventDefault()
                setIsDragging(true)
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={(e) => {
                e.preventDefault()
                setIsDragging(false)
                handleFiles(e.dataTransfer.files)
              }}
              className={`flex flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed px-6 py-14 text-center transition-colors ${
                isDragging ? "border-primary bg-accent" : "border-border"
              }`}
            >
              <div className="flex size-12 items-center justify-center rounded-full bg-accent text-primary">
                {isUploading ? <Loader2 className="size-6 animate-spin" /> : <UploadCloud className="size-6" />}
              </div>
              <div>
                <p className="font-medium text-foreground">Upload Medical Document</p>
                <p className="text-sm text-muted-foreground">PDF, DOCX, or TXT</p>
              </div>
              <Button onClick={() => fileInputRef.current?.click()} disabled={isUploading}>
                {isUploading ? "Uploading…" : "Choose File"}
              </Button>
              <p className="text-xs text-muted-foreground">or drag and drop</p>
              <input
                ref={fileInputRef}
                type="file"
                accept={ACCEPTED_EXTENSIONS.join(",")}
                className="hidden"
                onChange={(e) => handleFiles(e.target.files)}
              />
              {uploadError && (
                <div className="mt-2 flex items-center gap-2 rounded-md bg-status-danger-bg px-3 py-2 text-sm text-status-danger">
                  <AlertCircle className="size-4 shrink-0" />
                  {uploadError}
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Overview stat tiles */}
        <Card>
          <CardHeader>
            <CardTitle>Processing Overview</CardTitle>
          </CardHeader>
          <CardContent className="grid grid-cols-3 gap-3 text-center">
            <div>
              <p className="text-2xl font-semibold text-status-success">{processed}</p>
              <p className="text-xs text-muted-foreground">Processed</p>
            </div>
            <div>
              <p className="text-2xl font-semibold text-status-warning">{inProgress}</p>
              <p className="text-xs text-muted-foreground">In Progress</p>
            </div>
            <div>
              <p className="text-2xl font-semibold text-status-danger">{failed}</p>
              <p className="text-xs text-muted-foreground">Failed</p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Recent documents */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Documents</CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          {rows.length === 0 ? (
            <div className="flex flex-col items-center gap-2 px-6 py-12 text-center text-sm text-muted-foreground">
              <FileText className="size-8 text-muted-foreground/60" />
              No documents uploaded yet in this browser. Upload one above to get started.
            </div>
          ) : (
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="border-b border-border text-xs text-muted-foreground">
                  <th className="px-5 py-2.5 font-medium">Document Name</th>
                  <th className="px-5 py-2.5 font-medium">Uploaded</th>
                  <th className="px-5 py-2.5 font-medium">Status</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr
                    key={row.documentId}
                    onClick={() => openDocument(row.documentId)}
                    className="cursor-pointer border-b border-border last:border-0 hover:bg-muted/60"
                  >
                    <td className="flex items-center gap-2 px-5 py-3 font-medium text-foreground">
                      <FileText className="size-4 text-muted-foreground" />
                      {row.filename}
                    </td>
                    <td className="px-5 py-3 text-muted-foreground">{formatRelativeTime(row.uploadedAt)}</td>
                    <td className="px-5 py-3">
                      {row.document ? (
                        <Badge variant={statusBadgeVariant(row.document.processing_status)}>
                          {row.document.processing_status}
                        </Badge>
                      ) : (
                        <span className="text-xs text-muted-foreground">{row.loadError ?? "loading…"}</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
