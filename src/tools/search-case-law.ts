import type { Database } from '@ansvar/mcp-sqlite';
import { generateResponseMetadata, type ToolResponse } from '../utils/metadata.js';
import axios from 'axios';
import { Parser } from 'xml2js';
import {
  buildCitation,
  withCitationAttribution,
  type CitationMetadata,
} from '../utils/citation.js';

export interface SearchCaseLawInput {
  query: string;
  court?: string;
  ecli?: string;
  legal_domain?: string;
  procedure_type?: string;
  date_from?: string;
  date_to?: string;
  limit?: number;
}

export interface SearchCaseLawResult {
  document_id: string;
  document_title: string;
  ecli: string;
  court: string;
  case_number: string | null;
  decision_date: string | null;
  procedure_type: string | null;
  legal_domain: string | null;
  summary: string | null;
  snippet: string | null;
  relevance: number | null;
  url: string | null;
  _citation?: CitationMetadata;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const RECHTSPRAAK_SEARCH_URL = 'https://data.rechtspraak.nl/uitspraken/zoeken';
const RESULT_CAP = 3;

// ---------------------------------------------------------------------------
// Query optimisation
// ---------------------------------------------------------------------------

// Words that appear in almost every employment/civil-law sentence and add no
// discriminative signal when sent to the Rechtspraak full-text API.
const FILLER_WORDS = new Set([
  // Dutch function words
  'de',
  'het',
  'een',
  'van',
  'en',
  'in',
  'op',
  'te',
  'met',
  'voor',
  'dat',
  'die',
  'ook',
  'nog',
  'wel',
  'niet',
  'zijn',
  'heeft',
  'wordt',
  'kan',
  'moet',
  'naar',
  'over',
  'door',
  'maar',
  'dit',
  'dan',
  'om',
  'bij',
  'als',
  'uit',
  'aan',
  'tot',
  'zo',
  'er',
  'als',
  'aan',
  'was',
  // Domain-specific noise — ubiquitous in employment contexts, no search value
  'werknemer',
  'werkgever',
  'bedrijf',
  'kas',
  'omdat',
  'mijn',
  'medewerker',
  'collega',
  'baas',
  'chef',
  'zaak',
]);

// Ordered list of known Dutch legal multi-word phrases. Longer phrases are
// listed first so they match before any shorter sub-phrase would.
const LEGAL_PHRASES: readonly string[] = [
  'ontslag op staande voet',
  'kennelijk onredelijk ontslag',
  'arbeidsovereenkomst voor onbepaalde tijd',
  'onrechtmatige daad',
  'dringende reden',
  'wettelijke rente',
  'op staande voet',
];

interface PhraseExtractionResult {
  quoted: string[]; // phrase strings ready for the API, e.g. '"ontslag op staande voet"'
  remainder: string; // lowercased query with matched phrases removed
}

function extractLegalPhrases(query: string): PhraseExtractionResult {
  let remainder = query.toLowerCase();
  const quoted: string[] = [];
  for (const phrase of LEGAL_PHRASES) {
    if (remainder.includes(phrase)) {
      quoted.push(`"${phrase}"`);
      remainder = remainder.replace(phrase, ' ');
    }
  }
  return { quoted, remainder };
}

/**
 * Build an optimised API query for data.rechtspraak.nl.
 *
 * Queries with ≤ 3 words are passed through unchanged.
 * Longer queries are reduced to: quoted legal phrases + up to 3 remaining
 * meaningful keywords, with filler words stripped.
 *
 * Exported for unit-testing.
 */
export function buildApiQuery(rawQuery: string): string {
  const words = rawQuery.trim().split(/\s+/);
  if (words.length <= 3) return rawQuery;

  const { quoted, remainder } = extractLegalPhrases(rawQuery);

  const extraTokens = remainder
    .split(/\s+/)
    .map((w) => w.replace(/[^\p{L}]/gu, ''))
    .filter((w) => w.length > 2 && !FILLER_WORDS.has(w.toLowerCase()));

  const slotsLeft = Math.max(0, 3 - quoted.length);
  const parts = [...quoted, ...extraTokens.slice(0, slotsLeft)];

  // Always return something — fall back to first 3 original words if cleaning
  // removed everything.
  return parts.length > 0 ? parts.join(' ') : words.slice(0, 3).join(' ');
}

/**
 * Derive a single-phrase fallback query for when the primary returns 0 hits.
 * Prefers the first detected legal phrase; otherwise the first substantive token.
 */
function buildFallbackQuery(rawQuery: string): string | null {
  const { quoted, remainder } = extractLegalPhrases(rawQuery);
  if (quoted.length > 0) return quoted[0];

  const token = remainder
    .split(/\s+/)
    .map((w) => w.replace(/[^\p{L}]/gu, ''))
    .find((w) => w.length > 3 && !FILLER_WORDS.has(w.toLowerCase()));

  return token ?? null;
}

// Known ECLI court codes → display names. Prefix-based fallback handles the rest.
const COURT_CODE_MAP: Record<string, string> = {
  HR: 'Hoge Raad',
  RVS: 'Raad van State',
  CRVB: 'Centrale Raad van Beroep',
  CBB: 'College van Beroep voor het bedrijfsleven',
  GHAMS: 'Gerechtshof Amsterdam',
  GHSHE: "Gerechtshof 's-Hertogenbosch",
  GHARL: 'Gerechtshof Arnhem-Leeuwarden',
  GHDHA: 'Gerechtshof Den Haag',
};

// Courts that hold the most precedential authority for SME legal questions.
const HIGH_AUTHORITY_COURTS = ['hoge raad', 'gerechtshof'];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function ecliToUrl(ecli: string | null): string | null {
  if (ecli) return `https://uitspraken.rechtspraak.nl/details?id=${encodeURIComponent(ecli)}`;
  return null;
}

function extractEcliFromText(text: string): string {
  const match = text.match(/ECLI:[A-Z]{2}:[A-Z0-9]+:\d{4}:\w+/i);
  return match ? match[0].toUpperCase() : '';
}

function courtCodeToName(code: string): string {
  const upper = code.toUpperCase();
  if (COURT_CODE_MAP[upper]) return COURT_CODE_MAP[upper];
  if (upper.startsWith('GH')) return 'Gerechtshof';
  if (upper.startsWith('RB')) return 'Rechtbank';
  return upper;
}

function extractCourtFromEcli(ecli: string): string {
  const parts = ecli.split(':');
  return parts.length >= 3 ? courtCodeToName(parts[2]) : ecli;
}

// ---------------------------------------------------------------------------
// Citations (kept intact — same logic as the original)
// ---------------------------------------------------------------------------

function addResultCitations(rows: SearchCaseLawResult[]): SearchCaseLawResult[] {
  return rows.map((row) => {
    const canonicalRef = row.ecli || row.document_id;
    const displayText = [row.court, row.decision_date, canonicalRef].filter(Boolean).join(' ');
    const lookupArgs: Record<string, string> = row.ecli
      ? { ecli: row.ecli }
      : { document_id: row.document_id };
    const citation = buildCitation(
      canonicalRef,
      displayText,
      'search_case_law',
      lookupArgs,
      row.url || ecliToUrl(row.ecli),
      [row.document_id, row.case_number, row.document_title].filter((value): value is string =>
        Boolean(value),
      ),
    );

    return {
      ...row,
      _citation: withCitationAttribution(citation, {
        jurisdiction: 'NL',
        source: row.court || 'rechtspraak.nl',
        article: canonicalRef,
        publisher: 'De Rechtspraak (Dutch Judiciary)',
        license: 'Public-Domain',
        effective_date: row.decision_date,
      }),
    };
  });
}

// ---------------------------------------------------------------------------
// Authority + recency sort
// ---------------------------------------------------------------------------

function ecliYear(ecli: string): number {
  const parts = ecli.split(':');
  const year = parts.length >= 4 ? parseInt(parts[3], 10) : 0;
  return isNaN(year) ? 0 : year;
}

function sortByAuthority(results: SearchCaseLawResult[]): SearchCaseLawResult[] {
  return [...results].sort((a, b) => {
    // Primary: most recent first
    const yearDiff = ecliYear(b.ecli) - ecliYear(a.ecli);
    if (yearDiff !== 0) return yearDiff;
    // Tiebreaker within same year: higher-authority court first
    const aHigh = HIGH_AUTHORITY_COURTS.some((c) => a.court.toLowerCase().includes(c));
    const bHigh = HIGH_AUTHORITY_COURTS.some((c) => b.court.toLowerCase().includes(c));
    if (aHigh && !bHigh) return -1;
    if (!aHigh && bHigh) return 1;
    return 0;
  });
}

// ---------------------------------------------------------------------------
// Live fetch from data.rechtspraak.nl (Atom XML feed)
// ---------------------------------------------------------------------------

// xml2js with explicitArray:false parses <el attr="x">text</el> as { $: { attr }, _: 'text' }
// and <el attr="x" /> (no text) as { $: { attr } }. These helpers normalise both cases.
function xmlText(el: unknown): string {
  if (typeof el === 'string') return el;
  if (el && typeof el === 'object') {
    const v = (el as Record<string, unknown>)['_'];
    if (typeof v === 'string') return v;
  }
  return '';
}

function xmlLinkHref(el: unknown): string {
  if (typeof el === 'string') return el;
  if (el && typeof el === 'object') {
    const attrs = (el as Record<string, unknown>)['$'] as Record<string, unknown> | undefined;
    if (typeof attrs?.['href'] === 'string') return attrs['href'];
  }
  return '';
}

// Fetch the inhoudsindicatie (case summary) from the full case XML document.
// Runs in parallel for each result; failures are silently ignored.
async function fetchCaseContent(ecli: string): Promise<string | null> {
  try {
    const url = `https://data.rechtspraak.nl/uitspraken/content?id=${encodeURIComponent(ecli)}`;
    const response = await axios.get<string>(url, { timeout: 5000, responseType: 'text' });
    const match = response.data.match(/<inhoudsindicatie[^>]*>([\s\S]*?)<\/inhoudsindicatie>/i);
    if (match) {
      return match[1]
        .replace(/<[^>]+>/g, ' ')
        .replace(/\s+/g, ' ')
        .trim()
        .slice(0, 600);
    }
    return null;
  } catch {
    return null;
  }
}

async function fetchFromRechtspraak(query: string): Promise<SearchCaseLawResult[]> {
  const searchUrl = `${RECHTSPRAAK_SEARCH_URL}?return=DOC&q=${encodeURIComponent(query)}&max=25&sort=DESC`;
  const response = await axios.get<string>(searchUrl, { timeout: 8000, responseType: 'text' });

  const xmlParser = new Parser({ explicitArray: false });
  const parsedXml = (await xmlParser.parseStringPromise(response.data)) as Record<string, unknown>;

  const feed = parsedXml['feed'] as Record<string, unknown> | undefined;
  const entries = feed?.['entry'];
  const rawCases = entries ? (Array.isArray(entries) ? entries : [entries]) : [];

  console.log(`[Rechtspraak API Live Hits]: ${rawCases.length}`);

  return rawCases.map((item: unknown) => {
    const e = item as Record<string, unknown>;

    const idRaw = xmlText(e['id']);
    const titleRaw = xmlText(e['title']);
    const summaryRaw = xmlText(e['summary']) || xmlText(e['content']);
    const updatedRaw = xmlText(e['updated']);
    const linkHref = xmlLinkHref(e['link']);

    const ecli = extractEcliFromText(idRaw) || extractEcliFromText(titleRaw);
    const court = ecli ? extractCourtFromEcli(ecli) : '';
    const resolvedUrl = linkHref || ecliToUrl(ecli);

    return {
      document_id: ecli || idRaw || titleRaw,
      document_title: titleRaw,
      ecli: ecli || idRaw,
      court,
      case_number: null,
      decision_date: updatedRaw ? updatedRaw.slice(0, 10) : null,
      procedure_type: null,
      legal_domain: null,
      summary: summaryRaw || null,
      snippet: summaryRaw ? summaryRaw.slice(0, 200) + (summaryRaw.length > 200 ? '…' : '') : null,
      relevance: null,
      url: resolvedUrl,
    };
  });
}

// ---------------------------------------------------------------------------
// Exported tool handler
// ---------------------------------------------------------------------------

export async function searchCaseLaw(
  db: Database,
  input: SearchCaseLawInput,
): Promise<ToolResponse<SearchCaseLawResult[]>> {
  // ECLI direct lookup reuses the same query path — the API accepts ECLI strings as queries.
  const rawQuery = input.ecli ?? input.query ?? '';

  if (!rawQuery.trim()) {
    return { results: [], _metadata: generateResponseMetadata(db) };
  }

  const apiQuery = buildApiQuery(rawQuery);

  try {
    let raw = await fetchFromRechtspraak(apiQuery);

    if (raw.length === 0) {
      const fallback = buildFallbackQuery(rawQuery);
      if (fallback && fallback !== apiQuery) {
        raw = await fetchFromRechtspraak(fallback);
      }
    }

    const sorted = sortByAuthority(raw);
    const capped = sorted.slice(0, RESULT_CAP);

    // Enrich with full case content (parallel, fail-safe — empty summary means API returned "-")
    const enriched = await Promise.all(
      capped.map(async (result) => {
        if (!result.summary || result.summary === '-') {
          const content = await fetchCaseContent(result.ecli);
          if (content) {
            return {
              ...result,
              summary: content,
              snippet: content.slice(0, 200) + (content.length > 200 ? '…' : ''),
            };
          }
        }
        return result;
      }),
    );

    const results = addResultCitations(enriched);

    return {
      results,
      _metadata: generateResponseMetadata(db),
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return {
      results: [],
      _metadata: {
        ...generateResponseMetadata(db),
        note: `Rechtspraak.nl is tijdelijk niet bereikbaar (${message}). Probeer het later opnieuw.`,
      },
    };
  }
}
