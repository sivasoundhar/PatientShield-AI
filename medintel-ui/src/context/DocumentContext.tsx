import { createContext, useContext, useEffect, useState, type ReactNode } from "react"

/**
 * Tracks the "current" document the user is working with, so sidebar nav
 * links to Processing/Workspace/Chat/Audit can jump straight to it instead
 * of forcing a document picker on every navigation. Persisted to
 * localStorage (survives a page refresh) — this is UI convenience state
 * only, never a source of truth for document data itself (that always
 * comes from the API).
 */
interface DocumentContextValue {
  currentDocumentId: string | null
  setCurrentDocumentId: (id: string | null) => void
}

const STORAGE_KEY = "patientshield.currentDocumentId"

const DocumentContext = createContext<DocumentContextValue | undefined>(undefined)

export function DocumentProvider({ children }: { children: ReactNode }) {
  const [currentDocumentId, setCurrentDocumentIdState] = useState<string | null>(() =>
    localStorage.getItem(STORAGE_KEY),
  )

  useEffect(() => {
    if (currentDocumentId) {
      localStorage.setItem(STORAGE_KEY, currentDocumentId)
    } else {
      localStorage.removeItem(STORAGE_KEY)
    }
  }, [currentDocumentId])

  return (
    <DocumentContext.Provider value={{ currentDocumentId, setCurrentDocumentId: setCurrentDocumentIdState }}>
      {children}
    </DocumentContext.Provider>
  )
}

/** Use when: any page/component needs to read or set the active document id. */
export function useCurrentDocument() {
  const ctx = useContext(DocumentContext)
  if (!ctx) throw new Error("useCurrentDocument must be used within a DocumentProvider")
  return ctx
}
