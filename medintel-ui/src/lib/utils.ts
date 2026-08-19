import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

/** Merge Tailwind classes with conditional logic, resolving conflicting utility classes correctly. */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}
