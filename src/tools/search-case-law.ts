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

  // Sort by descending word length — longer words tend to be more specific (e.g.
  // "diefstal" beats "voet"). When a phrase was extracted, 2 extra slots remain;
  // without a phrase, allow up to 4 to avoid dropping key discriminating terms.
  const extraTokens = remainder
    .split(/\s+/)
    .map((w) => w.replace(/[^\p{L}]/gu, ''))
    .filter((w) => w.length > 2 && !FILLER_WORDS.has(w.toLowerCase()))
    .sort((a, b) => b.length - a.length);

  const maxSlots = quoted.length > 0 ? 3 : 4;
  const slotsLeft = Math.max(0, maxSlots - quoted.length);
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
// Local relevance scoring + sort
// ---------------------------------------------------------------------------

function ecliYear(ecli: string): number {
  const parts = ecli.split(':');
  const year = parts.length >= 4 ? parseInt(parts[3], 10) : 0;
  return isNaN(year) ? 0 : year;
}

// Markers that indicate a purely criminal-law case — penalised when query is civil/labour.
const CRIMINAL_MARKERS =
  /gevangenisstraf|tenlastelegging|openbaar ministerie|opiumwet|verdachte\b|politierechter|vrijspraak|vrijgesproken|wederrechtelijke vrijheidsberoving|bedreiging met geweld|schuldigverklaring/i;

// Markers that indicate administrative law — penalised for civil/labour queries.
const ADMIN_MARKERS =
  /hoorzitting bezwaar|bezwaarschrift|bestuursorgaan|awb\b|algemene wet bestuursrecht|beroepschrift|bestuursrechter|bezwaarprocedure/i;

// Markers that indicate employment law content — rewarded.
const EMPLOYMENT_MARKERS =
  /ontslag|werknemer|werkgever|arbeidsovereenkomst|dienstverband|loon|cao\b/i;

function extractScoringTerms(query: string): string[] {
  return query
    .toLowerCase()
    .split(/\s+/)
    .map((w) => w.replace(/[^\p{L}]/gu, ''))
    .filter((w) => w.length > 3 && !FILLER_WORDS.has(w));
}

function scoreRelevance(result: SearchCaseLawResult, terms: string[]): number {
  const haystack = `${result.document_title} ${result.summary ?? ''}`.toLowerCase();
  let score = terms.reduce((s, term) => s + (haystack.includes(term) ? 1 : 0), 0);
  if (CRIMINAL_MARKERS.test(haystack)) score -= 3;
  if (ADMIN_MARKERS.test(haystack)) score -= 2;
  if (EMPLOYMENT_MARKERS.test(haystack)) score += 1;
  // Prefer recent cases (year 2015+ gets a small boost)
  const yr = ecliYear(result.ecli);
  if (yr >= 2020) score += 2;
  else if (yr >= 2015) score += 1;
  return score;
}

function sortByRelevance(results: SearchCaseLawResult[], terms: string[]): SearchCaseLawResult[] {
  return [...results].sort((a, b) => {
    const scoreDiff = scoreRelevance(b, terms) - scoreRelevance(a, terms);
    if (scoreDiff !== 0) return scoreDiff;
    // Within same score: higher authority first
    const aHigh = HIGH_AUTHORITY_COURTS.some((c) => a.court.toLowerCase().includes(c));
    const bHigh = HIGH_AUTHORITY_COURTS.some((c) => b.court.toLowerCase().includes(c));
    if (aHigh && !bHigh) return -1;
    if (!aHigh && bHigh) return 1;
    // Within same authority: most recent first
    return ecliYear(b.ecli) - ecliYear(a.ecli);
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
        .slice(0, 1500);
    }
    return null;
  } catch {
    return null;
  }
}

async function fetchFromRechtspraak(
  query: string,
  dateFrom?: string,
  _dateTo?: string,
): Promise<SearchCaseLawResult[]> {
  // modified= filters by indexing date, excluding old AA-series cases (pre-2010) that
  // have no inhoudsindicatie XML and score 0. Callers can override with explicit dateFrom.
  const effectiveDateFrom = dateFrom ?? '2015-01-01';
  // No sort parameter → API defaults to relevance ranking, avoiding recency bias.
  // type=uitspraak filters out PHR conclusions, which are not court decisions.
  let searchUrl = `${RECHTSPRAAK_SEARCH_URL}?return=DOC&q=${encodeURIComponent(query)}&max=25&type=uitspraak`;
  searchUrl += `&modified=${effectiveDateFrom}`;
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

    // <updated> is the Atom indexing date, not the decision date.
    // For old digitized cases (e.g. 1998 case indexed in 2013) this would mislead.
    // Trust <updated> only when its year matches the ECLI year; otherwise derive from ECLI.
    const ecliYr = ecli ? ecliYear(ecli) : 0;
    const updatedYear = updatedRaw ? parseInt(updatedRaw.slice(0, 4), 10) : 0;
    const decisionDate =
      updatedRaw && Math.abs(updatedYear - ecliYr) <= 1
        ? updatedRaw.slice(0, 10)
        : ecliYr > 0
          ? `${ecliYr}-01-01`
          : null;

    return {
      document_id: ecli || idRaw || titleRaw,
      document_title: titleRaw,
      ecli: ecli || idRaw,
      court,
      case_number: null,
      decision_date: decisionDate,
      procedure_type: null,
      legal_domain: null,
      summary: summaryRaw || null,
      snippet: summaryRaw ? summaryRaw.slice(0, 200) + (summaryRaw.length > 200 ? '…' : '') : null,
      relevance: null,
      url: resolvedUrl,
    };
  });
}

// Maps common legal_domain values to a keyword appended to the API query.
// This scopes results to the right area of law when the field is provided.
const DOMAIN_KEYWORDS: Record<string, string> = {
  arbeidsrecht: 'arbeidsrecht',
  civielrecht: 'civielrecht',
  'civiel recht': 'civielrecht',
  bestuursrecht: 'bestuursrecht',
  strafrecht: 'strafrecht',
  ondernemingsrecht: 'ondernemingsrecht',
  huurrecht: 'huurrecht',
  familierecht: 'familierecht',
  verbintenissenrecht: 'verbintenissenrecht',
};

function domainKeyword(legal_domain: string | undefined): string | null {
  if (!legal_domain) return null;
  const key = legal_domain.toLowerCase().trim();
  return DOMAIN_KEYWORDS[key] ?? null;
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

  let apiQuery = buildApiQuery(rawQuery);

  // Append legal domain keyword so the API scopes to the right area of law.
  const dk = domainKeyword(input.legal_domain);
  if (dk) apiQuery = `${apiQuery} ${dk}`;

  try {
    let raw = await fetchFromRechtspraak(apiQuery, input.date_from, input.date_to);

    if (raw.length === 0) {
      const fallback = buildFallbackQuery(rawQuery);
      if (fallback && fallback !== apiQuery) {
        raw = await fetchFromRechtspraak(fallback, input.date_from, input.date_to);
      }
    }

    // Enrich ALL candidates with the official inhoudsindicatie before scoring.
    // Always prefer fetched content over the Atom feed summary (which is often '-' or a short stub).
    // All fetches run in parallel — wall time is bounded by the 5 s timeout, not by count.
    const enriched = await Promise.all(
      raw.map(async (result: SearchCaseLawResult) => {
        const content = await fetchCaseContent(result.ecli);
        if (content) {
          return {
            ...result,
            summary: content,
            snippet: content.slice(0, 200) + (content.length > 200 ? '…' : ''),
          };
        }
        return result;
      }),
    );

    // Score every enriched candidate and emit a log line for each so scores are visible.
    const scoringTerms = extractScoringTerms(rawQuery);
    enriched.forEach((result) => {
      const score = scoreRelevance(result, scoringTerms);
      const hasContent = !!(result.summary && result.summary !== '-' && result.summary.length > 20);
      const matched = scoringTerms.filter((t) =>
        `${result.document_title} ${result.summary ?? ''}`.toLowerCase().includes(t),
      );
      console.error(
        `[score] ${result.ecli} → ${score} | content=${hasContent ? 'YES' : 'NO'} | terms=[${matched.join(', ')}]`,
      );
    });

    // Sort by score; prefer results with at least one term match.
    // Fallback uses top half of scored set — avoids pulling in clearly irrelevant cases.
    const sorted = sortByRelevance(enriched, scoringTerms);
    const qualified = sorted.filter((r) => scoreRelevance(r, scoringTerms) >= 1);
    const pool = qualified.length > 0 ? qualified : sorted.slice(0, Math.ceil(sorted.length / 2));
    const capped = pool.slice(0, RESULT_CAP);

    console.error(`[score] Selected top ${capped.length}: ${capped.map((r) => r.ecli).join(', ')}`);

    const results = addResultCitations(capped);

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
