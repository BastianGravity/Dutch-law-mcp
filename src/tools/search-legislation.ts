import type { Database } from '@ansvar/mcp-sqlite';
import axios from 'axios';
import { buildFtsQueryVariants } from '../utils/fts-query.js';
import { normalizeAsOfDate } from '../utils/as-of-date.js';
import { resolveDocumentId } from '../utils/document-id.js';
import { generateResponseMetadata, type ToolResponse } from '../utils/metadata.js';
import { withSqliteLockRetry } from '../utils/sqlite-retry.js';
import {
  buildProvisionCitation,
  withCitationAttribution,
  type CitationMetadata,
} from '../utils/citation.js';

export interface SearchLegislationInput {
  query: string;
  original_query?: string;
  document_id?: string;
  status?: string;
  as_of_date?: string;
  limit?: number;
}

export interface SearchLegislationResult {
  document_id: string;
  document_title: string;
  provision_ref: string;
  book: string | null;
  chapter: string | null;
  section: string | null;
  article: string;
  title: string | null;
  snippet: string;
  relevance: number;
  source_url: string | null;
  effective_date: string | null;
  valid_from?: string | null;
  valid_to?: string | null;
  _citation?: CitationMetadata;
}

const MAX_LIMIT = 50;
const DEFAULT_LIMIT = 10;

function clampLimit(limit: number | undefined): number {
  if (limit == null) return DEFAULT_LIMIT;
  return Math.max(1, Math.min(limit, MAX_LIMIT));
}

function runFtsSearch(
  db: Database,
  ftsQuery: string,
  documentId: string | undefined,
  status: string | undefined,
  limit: number,
): SearchLegislationResult[] {
  const conditions: string[] = [];
  const params: (string | number)[] = [];

  conditions.push('provisions_fts MATCH ?');
  params.push(ftsQuery);

  if (documentId) {
    conditions.push('p.document_id = ?');
    params.push(documentId);
  }

  if (status) {
    conditions.push('d.status = ?');
    params.push(status);
  }

  const whereClause = conditions.length > 0 ? `WHERE ${conditions.join(' AND ')}` : '';

  const sql = `
    SELECT
      p.document_id,
      d.title AS document_title,
      p.provision_ref,
      p.book,
      p.chapter,
      p.section,
      p.article,
      p.title,
      snippet(provisions_fts, 0, '**', '**', '...', 32) AS snippet,
      bm25(provisions_fts) AS relevance,
      COALESCE(d.url, 'https://wetten.overheid.nl/' || p.document_id) AS source_url,
      d.in_force_date AS effective_date
    FROM provisions_fts
    JOIN legal_provisions AS p ON provisions_fts.rowid = p.id
    JOIN legal_documents AS d ON p.document_id = d.id
    ${whereClause}
    ORDER BY (bm25(provisions_fts) * CASE
      WHEN d.title LIKE '%Burgerlijk Wetboek%'                           THEN 5.0
      WHEN d.title LIKE '%Wetboek van Strafrecht%'                       THEN 5.0
      WHEN d.title LIKE '%Wetboek van Burgerlijke Rechtsvordering%'      THEN 5.0
      WHEN d.title LIKE '%Wetboek van Strafvordering%'                   THEN 5.0
      WHEN d.title LIKE '%Algemene wet bestuursrecht%'                   THEN 4.0
      WHEN d.title LIKE '%Arbeidsomstandighedenwet%'                     THEN 3.0
      WHEN d.title LIKE '%Wet op het financieel toezicht%'               THEN 3.0
      WHEN d.title LIKE 'Invoeringswet%'                                 THEN 0.3
      WHEN d.title LIKE 'Aanpassingswet%'                                THEN 0.3
      WHEN d.title LIKE 'Verzamelwet%'                                   THEN 0.3
      WHEN d.title LIKE 'Wijzigingswet%'                                 THEN 0.3
      ELSE 1.0
    END)
    LIMIT ?
  `;
  params.push(limit);

  return db.prepare(sql).all(...params) as SearchLegislationResult[];
}

function runVersionedFtsSearch(
  db: Database,
  ftsQuery: string,
  asOfDate: string,
  documentId: string | undefined,
  status: string | undefined,
  limit: number,
): SearchLegislationResult[] {
  const conditions: string[] = [];
  const params: (string | number)[] = [];

  conditions.push('provision_versions_fts MATCH ?');
  params.push(ftsQuery);

  conditions.push('(pv.valid_from IS NULL OR pv.valid_from <= ?)');
  params.push(asOfDate);

  conditions.push('(pv.valid_to IS NULL OR pv.valid_to > ?)');
  params.push(asOfDate);

  if (documentId) {
    conditions.push('pv.document_id = ?');
    params.push(documentId);
  }

  if (status) {
    conditions.push('d.status = ?');
    params.push(status);
  }

  const whereClause = conditions.length > 0 ? `WHERE ${conditions.join(' AND ')}` : '';

  const sql = `
    SELECT
      pv.document_id,
      d.title AS document_title,
      pv.provision_ref,
      pv.book,
      pv.chapter,
      pv.section,
      pv.article,
      pv.title,
      snippet(provision_versions_fts, 0, '**', '**', '...', 32) AS snippet,
      bm25(provision_versions_fts) AS relevance,
      COALESCE(d.url, 'https://wetten.overheid.nl/' || pv.document_id) AS source_url,
      COALESCE(pv.valid_from, d.in_force_date) AS effective_date,
      pv.valid_from,
      pv.valid_to
    FROM provision_versions_fts
    JOIN legal_provision_versions AS pv ON provision_versions_fts.rowid = pv.id
    JOIN legal_documents AS d ON pv.document_id = d.id
    ${whereClause}
    ORDER BY (bm25(provision_versions_fts) * CASE
      WHEN d.title LIKE '%Burgerlijk Wetboek%'                           THEN 5.0
      WHEN d.title LIKE '%Wetboek van Strafrecht%'                       THEN 5.0
      WHEN d.title LIKE '%Wetboek van Burgerlijke Rechtsvordering%'      THEN 5.0
      WHEN d.title LIKE '%Wetboek van Strafvordering%'                   THEN 5.0
      WHEN d.title LIKE '%Algemene wet bestuursrecht%'                   THEN 4.0
      WHEN d.title LIKE '%Arbeidsomstandighedenwet%'                     THEN 3.0
      WHEN d.title LIKE '%Wet op het financieel toezicht%'               THEN 3.0
      WHEN d.title LIKE 'Invoeringswet%'                                 THEN 0.3
      WHEN d.title LIKE 'Aanpassingswet%'                                THEN 0.3
      WHEN d.title LIKE 'Verzamelwet%'                                   THEN 0.3
      WHEN d.title LIKE 'Wijzigingswet%'                                 THEN 0.3
      ELSE 1.0
    END)
    LIMIT ?
  `;
  params.push(limit);

  return db.prepare(sql).all(...params) as SearchLegislationResult[];
}

/**
 * Deduplicate search results by document_title + provision_ref.
 * Keeps the first (highest-ranked) occurrence.
 */
function deduplicateResults(
  rows: SearchLegislationResult[],
  limit: number,
): SearchLegislationResult[] {
  const seen = new Set<string>();
  const deduped: SearchLegislationResult[] = [];
  for (const row of rows) {
    const key = `${row.document_title}::${row.provision_ref}`;
    if (seen.has(key)) continue;
    seen.add(key);
    deduped.push(row);
    if (deduped.length >= limit) break;
  }
  return deduped;
}

function buildWettenUrl(documentId: string, article: string | null | undefined): string {
  const base = `https://wetten.overheid.nl/jci1.3:c:${documentId}`;
  return article ? `${base}&artikel=${article}` : base;
}

function addResultCitations(rows: SearchLegislationResult[]): SearchLegislationResult[] {
  return rows.map((row) => {
    const articleUrl = buildWettenUrl(row.document_id, row.article);
    const citation = buildProvisionCitation(
      row.document_id,
      row.document_title || '',
      row.provision_ref || row.article || '',
      row.document_id,
      row.provision_ref || row.article || '',
      articleUrl,
      null,
    );

    return {
      ...row,
      source_url: articleUrl,
      _citation: withCitationAttribution(citation, {
        jurisdiction: 'NL',
        source: row.document_id,
        source_full_name: row.document_title,
        article: row.provision_ref || row.article,
        publisher: 'Dutch Government (wetten.overheid.nl)',
        license: 'Public-Domain',
        effective_date: row.effective_date || row.valid_from || null,
      }),
    };
  });
}

// ---------------------------------------------------------------------------
// STAGE 0 — Haiku generates targeted legal search terms for FTS
// ---------------------------------------------------------------------------

const HAIKU_TERMS_SYSTEM_PROMPT = `Jij bent een juridisch zoeksysteem voor de Nederlandse wetgevingsdatabase (SQLite FTS).

TAAK:
Analyseer de juridische vraag en geef 3 tot 5 specifieke juridische termen of korte zinsdelen die letterlijk in Nederlandse wetsartikelen voorkomen en het meest relevant zijn voor deze vraag.

REGELS:
- Geef alleen termen die in wetteksten voorkomen (niet in omgangstaal)
- Één term per regel
- Geen uitleg, geen nummers, geen bullets
- Maximaal 5 termen

VOORBEELDEN:

Input: "Ik heb een werknemer die regelmatig te laat komt en slecht presteert, kan ik hem ontslaan?"
Output:
disfunctioneren
ontbinding arbeidsovereenkomst
verbetertraject
herplaatsing

Input: "Klant betaalt niet, kan ik rente en incassokosten in rekening brengen?"
Output:
verzuim
wettelijke rente
buitengerechtelijke incassokosten
handelsovereenkomst

Input: "Gekocht product werkt niet, wat zijn mijn rechten?"
Output:
non-conformiteit
aflevering
herstel vervanging
ontbinding koopovereenkomst

Input: "Mijn werknemer is ziek, hoelang moet ik loon doorbetalen?"
Output:
loondoorbetaling ziekte
arbeidsongeschiktheid
re-integratie
passende arbeid`;

async function buildFtsTermsWithHaiku(query: string): Promise<string[]> {
  const model = process.env.ANTHROPIC_TITLE_MODEL;
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!model || !apiKey) return [];

  try {
    const response = await axios.post<{ content: Array<{ text: string }> }>(
      'https://api.anthropic.com/v1/messages',
      {
        model,
        max_tokens: 120,
        temperature: 0,
        system: HAIKU_TERMS_SYSTEM_PROMPT,
        messages: [{ role: 'user', content: query }],
      },
      {
        headers: {
          'x-api-key': apiKey,
          'anthropic-version': '2023-06-01',
          'content-type': 'application/json',
        },
        timeout: 6000,
      },
    );

    const text = response.data?.content?.[0]?.text?.trim() ?? '';
    const terms = text
      .split(/\n/)
      .map((l) => l.trim())
      .filter((l) => l.length > 2 && l.length < 60);
    console.error(`[haiku-terms] "${query.slice(0, 60)}" → ${terms.join(' | ') || 'none'}`);
    return terms;
  } catch (err) {
    console.error(`[haiku-terms] failed (${String(err)})`);
    return [];
  }
}

export async function searchLegislation(
  db: Database,
  input: SearchLegislationInput,
): Promise<ToolResponse<SearchLegislationResult[]>> {
  const { query, status } = input;
  const asOfDate = normalizeAsOfDate(input.as_of_date);
  const limit = clampLimit(input.limit);
  // Fetch extra rows to account for deduplication
  const fetchLimit = limit * 2;

  const variants = buildFtsQueryVariants(query);
  if (variants.tooBroad) {
    return {
      results: [],
      _metadata: {
        ...generateResponseMetadata(db),
        note:
          `Query too broad: "${query}" contains only common Dutch words. ` +
          'Please provide at least one specific legal term ' +
          '(e.g. "onrechtmatige daad" instead of "wet").',
      },
    };
  }
  if (!variants.primary) {
    return { results: [], _metadata: generateResponseMetadata(db) };
  }
  const primaryQuery = variants.primary;
  const fallbackQuery = variants.fallback;

  // Resolve document_id from title if provided
  let resolvedDocId: string | undefined;
  if (input.document_id) {
    const resolved = resolveDocumentId(db, input.document_id);
    resolvedDocId = resolved ?? undefined;
    if (!resolved) {
      return {
        results: [],
        _metadata: {
          ...generateResponseMetadata(db),
          note: `No document found matching "${input.document_id}"`,
        },
      };
    }
  }

  const search = asOfDate
    ? (q: string) => runVersionedFtsSearch(db, q, asOfDate, resolvedDocId, status, fetchLimit)
    : (q: string) => runFtsSearch(db, q, resolvedDocId, status, fetchLimit);

  // Run Haiku term generation and primary FTS in parallel
  const haikuInput = input.original_query?.trim() || query;
  const [haikuTerms, primaryResults] = await Promise.all([
    buildFtsTermsWithHaiku(haikuInput),
    withSqliteLockRetry(() => search(primaryQuery)),
  ]);

  // Run FTS for each Haiku term (sequential to avoid DB lock contention)
  const haikuResults: SearchLegislationResult[] = [];
  for (const term of haikuTerms) {
    const rows = await withSqliteLockRetry(() => search(term));
    haikuResults.push(...rows);
  }

  let broadened = false;
  let ftsBase = primaryResults;
  if (ftsBase.length === 0 && fallbackQuery) {
    ftsBase = await withSqliteLockRetry(() => search(fallbackQuery));
    broadened = ftsBase.length > 0;
  }

  // Merge: Haiku-guided FTS hits first (more targeted), primary FTS fills remaining slots
  const seenKeys = new Set<string>();
  const merged: SearchLegislationResult[] = [];

  for (const row of [...haikuResults, ...ftsBase]) {
    const key = `${row.document_id}::${row.provision_ref}`;
    if (!seenKeys.has(key)) {
      seenKeys.add(key);
      merged.push(row);
    }
  }

  const deduped = addResultCitations(deduplicateResults(merged, limit));

  return {
    results: deduped,
    _metadata: {
      ...generateResponseMetadata(db),
      ...(broadened ? { query_strategy: 'broadened' } : {}),
      ...(haikuTerms.length > 0 ? { haiku_terms: haikuTerms.length } : {}),
    },
  };
}
