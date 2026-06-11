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

const SEARCH_URL = 'https://data.rechtspraak.nl/uitspraken/zoeken';
const RESULT_CAP = 7;

// ---------------------------------------------------------------------------
// STAGE 1 — Query sanitization
// ---------------------------------------------------------------------------

const FILLER_WORDS = new Set([
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
  'was',
  'ik',
  'mijn',
  'wij',
  'hij',
  'zij',
  'haar',
  'hun',
  'ons',
  'wat',
  'hoe',
  'wie',
  'werknemer',
  'werkgever',
  'bedrijf',
  'omdat',
  'medewerker',
  'collega',
  'baas',
  'chef',
  'zaak',
  'situatie',
  'vraag',
]);

const LEGAL_PHRASES: readonly string[] = [
  'ontslag op staande voet',
  'kennelijk onredelijk ontslag',
  'arbeidsovereenkomst voor onbepaalde tijd',
  'onrechtmatige daad',
  'dringende reden',
  'wettelijke rente',
  'op staande voet',
  'ernstig verwijtbaar',
  'billijke vergoeding',
  'transitievergoeding',
  'verstoorde arbeidsverhouding',
  'twee maanden regel',
  'kort geding',
];

// OR-groups: if any entry from a group appears in the query, the whole group is added as (a OR b OR c)
const SYNONYM_GROUPS: ReadonlyArray<ReadonlyArray<string>> = [
  ['diefstal', 'fraude', 'verduistering', 'ontvreemding'],
  ['ziek', 'arbeidsongeschikt', 'ziekte', 'ziekmelding'],
  ['geweld', 'agressie', 'mishandeling', 'bedreiging'],
  ['feest', 'terras', 'stappen', 'vakantie', 'social'],
  ['alcohol', 'drank', 'drugs', 'intoxicatie'],
  ['discriminatie', 'pesten', 'intimidatie', 'racisme'],
  ['concurrentiebeding', 'relatiebeding', 'geheimhouding'],
  ['schade', 'schadevergoeding', 'aansprakelijkheid'],
];

function matchesSynonymGroup(term: string, group: ReadonlyArray<string>): boolean {
  return group.some(
    (s) =>
      s === term ||
      (s.length >= 4 && term.startsWith(s)) ||
      (term.length >= 4 && s.startsWith(term)),
  );
}

function extractLegalPhrases(query: string): { quoted: string[]; remainder: string } {
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

export function buildApiQuery(rawQuery: string): string {
  const words = rawQuery.trim().split(/\s+/);
  if (words.length <= 2) return rawQuery;

  const { quoted, remainder } = extractLegalPhrases(rawQuery);

  const keyTerms = remainder
    .split(/\s+/)
    .map((w) => w.replace(/[^\p{L}]/gu, '').toLowerCase())
    .filter((w) => w.length > 3 && !FILLER_WORDS.has(w));

  const used = new Set<string>();
  const queryParts: string[] = [...quoted];

  for (const term of keyTerms) {
    if (used.has(term)) continue;
    const group = SYNONYM_GROUPS.find((g) => matchesSynonymGroup(term, g));
    if (group) {
      queryParts.push(`(${group.join(' OR ')})`);
      group.forEach((s) => used.add(s));
    } else {
      queryParts.push(term);
      used.add(term);
    }
  }

  const limited = queryParts.slice(0, 6);
  if (limited.length === 0) return words.slice(0, 3).join(' ');

  // AND between all parts — quoted phrases + OR-synonym groups + single terms
  return limited.join(' AND ');
}

// ---------------------------------------------------------------------------
// STAGE 0 — Haiku transforms natural-language query to Boolean search syntax
// ---------------------------------------------------------------------------

async function buildBooleanQuery(rawQuery: string): Promise<string> {
  const model = process.env.ANTHROPIC_TITLE_MODEL;
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!model || !apiKey) return buildApiQuery(rawQuery);

  // The Rechtspraak API ignores AND/OR/NOT/parentheses — only quoted phrases and
  // space-separated implicit-AND work. Build: "quoted phrase" keyword1 keyword2
  const systemPrompt = `Je bent een juridisch zoeksysteem voor de Nederlandse Rechtspraak API.

HOE DE API WERKT (kritiek om te begrijpen):
- Spatie tussen termen = impliciete AND (alle termen moeten aanwezig zijn)
- "aanhalingstekens" = exacte woordvolgorde, werkt WEL
- AND / OR / NOT / haakjes worden GENEGEERD door de API — gebruik ze NOOIT
- Hoe meer specifieke termen, hoe preciezer de resultaten

JE TAAK:
Analyseer de juridische kern van de vraag en genereer een compacte zoekstring:
"primaire juridische term" aanvullend_feit1 aanvullend_feit2

REGELS:
1. Zet vaste juridische begrippen altijd tussen aanhalingstekens: "ontslag op staande voet", "dringende reden", "billijke vergoeding", "transitievergoeding"
2. Voeg maximaal 2 losse kernwoorden toe die het geval onderscheiden — NIET AND/OR/haakjes gebruiken
3. Kies kernwoorden die in de uitspraaktekst zullen voorkomen (juridische termen, niet omgangstaal)
4. Geef ALLEEN de zoekstring terug, geen uitleg

Voorbeelden:
Input: "werknemer meldt zich ziek via WhatsApp maar zit op terras"
Output: "ontslag op staande voet" ziekmelding terras

Input: "ontslagen wegens diefstal uit de kassa"
Output: "ontslag op staande voet" diefstal "dringende reden"

Input: "contract niet verlengd na zwangerschap"
Output: zwangerschap "arbeidsovereenkomst" discriminatie

Input: "transitievergoeding na 2 jaar ziekte"
Output: "transitievergoeding" arbeidsongeschikt langdurig

Input: "mijn baas schorst mij tijdens een disciplinair onderzoek"
Output: schorsing arbeidsovereenkomst disciplinair`;

  try {
    const response = await axios.post<{ content: Array<{ text: string }> }>(
      'https://api.anthropic.com/v1/messages',
      {
        model,
        max_tokens: 150,
        temperature: 0,
        system: systemPrompt,
        messages: [{ role: 'user', content: rawQuery }],
      },
      {
        headers: {
          'x-api-key': apiKey,
          'anthropic-version': '2023-06-01',
          'content-type': 'application/json',
        },
        timeout: 8000,
      },
    );

    const text = response.data?.content?.[0]?.text?.trim() ?? '';
    console.error(`[query-builder] "${rawQuery}" → "${text}"`);
    return text || buildApiQuery(rawQuery);
  } catch (err) {
    console.error(`[query-builder] failed (${String(err)}) — using deterministic fallback`);
    return buildApiQuery(rawQuery);
  }
}

function buildFallbackQuery(rawQuery: string): string | null {
  const { quoted, remainder } = extractLegalPhrases(rawQuery);
  if (quoted.length > 0) return quoted[0];
  const token = remainder
    .split(/\s+/)
    .map((w) => w.replace(/[^\p{L}]/gu, ''))
    .find((w) => w.length > 3 && !FILLER_WORDS.has(w.toLowerCase()));
  return token ?? null;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

function ecliToUrl(ecli: string | null): string | null {
  if (!ecli) return null;
  return `https://uitspraken.rechtspraak.nl/details?id=${encodeURIComponent(ecli)}`;
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

function ecliYear(ecli: string): number {
  const parts = ecli.split(':');
  const year = parts.length >= 4 ? parseInt(parts[3], 10) : 0;
  return isNaN(year) ? 0 : year;
}

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

// ---------------------------------------------------------------------------
// Citations
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
// STAGE 2 — Fetch from Rechtspraak
// ---------------------------------------------------------------------------

const DOMAIN_KEYWORDS: Record<string, string> = {
  arbeidsrecht: 'arbeidsrecht',
  civielrecht: 'civielrecht',
  'civiel recht': 'civielrecht',
  huurrecht: 'huurrecht',
  verbintenissenrecht: 'verbintenissenrecht',
  familierecht: 'familierecht',
  ondernemingsrecht: 'ondernemingsrecht',
  bestuursrecht: 'bestuursrecht',
  strafrecht: 'strafrecht',
};

async function fetchFromRechtspraak(
  query: string,
  legal_domain?: string,
  dateFrom?: string,
): Promise<SearchCaseLawResult[]> {
  // modified= filters by indexing date — excludes pre-2015 AA-series shells with no content.
  const cutoff = dateFrom ?? '2018-01-01';
  const domainKw = legal_domain
    ? (DOMAIN_KEYWORDS[legal_domain.toLowerCase().trim()] ?? null)
    : null;
  const finalQuery = domainKw ? `${query} ${domainKw}` : query;

  const url =
    `${SEARCH_URL}?return=DOC` +
    `&q=${encodeURIComponent(finalQuery)}` +
    `&max=20&type=Uitspraak` +
    `&modified=${cutoff}` +
    `&sort=DESC`;

  console.error(`[Rechtspraak] ${url}`);

  const response = await axios.get<string>(url, { timeout: 8000, responseType: 'text' });
  const xmlParser = new Parser({ explicitArray: false });
  const parsedXml = (await xmlParser.parseStringPromise(response.data)) as Record<string, unknown>;

  const feed = parsedXml['feed'] as Record<string, unknown> | undefined;
  const rawEntries = feed?.['entry'];
  const entries = rawEntries ? (Array.isArray(rawEntries) ? rawEntries : [rawEntries]) : [];

  console.error(`[Rechtspraak] raw entries: ${entries.length}`);

  return entries.map((item: unknown) => {
    const e = item as Record<string, unknown>;
    const idRaw = xmlText(e['id']);
    const titleRaw = xmlText(e['title']);
    const summaryRaw = xmlText(e['summary']) || xmlText(e['content']);
    const updatedRaw = xmlText(e['updated']);
    const linkHref = xmlLinkHref(e['link']);

    const ecli = extractEcliFromText(idRaw) || extractEcliFromText(titleRaw);
    const court = ecli ? extractCourtFromEcli(ecli) : '';
    const resolvedUrl = linkHref || ecliToUrl(ecli);

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

// ---------------------------------------------------------------------------
// STAGE 3 — Fetch official inhoudsindicatie (double-check: content must exist)
// ---------------------------------------------------------------------------

async function fetchInhoudsindicatie(ecli: string): Promise<string | null> {
  try {
    const url = `https://data.rechtspraak.nl/uitspraken/content?id=${encodeURIComponent(ecli)}`;
    const response = await axios.get<string>(url, { timeout: 5000, responseType: 'text' });
    const match = response.data.match(/<inhoudsindicatie[^>]*>([\s\S]*?)<\/inhoudsindicatie>/i);
    if (!match) return null;
    return (
      match[1]
        .replace(/<[^>]+>/g, ' ')
        .replace(/\s+/g, ' ')
        .trim()
        .slice(0, 2000) || null
    );
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// STAGE 3.5 — Keyword pre-filter: score all enriched cases, keep top 12 for Haiku
// ---------------------------------------------------------------------------

// Courts that never handle private employment law — filter by ECLI prefix (100% reliable)
const EXCLUDED_COURT_ECLI_PREFIXES = [
  'ECLI:NL:CRVB:', // Centrale Raad van Beroep — sociale zekerheid / ambtenaren
  'ECLI:NL:RVS:', // Raad van State — bestuursrecht
  'ECLI:NL:CBB:', // College van Beroep voor het bedrijfsleven
];

function isExcludedCourt(ecli: string): boolean {
  const upper = ecli.toUpperCase();
  return EXCLUDED_COURT_ECLI_PREFIXES.some((prefix) => upper.startsWith(prefix));
}

const CRIMINAL_MARKERS =
  /strafrecht|officier van justitie|gevangenisstraf|tenlastelegging|verdachte\b|openbaar ministerie|vrijspraak|opiumwet/i;

// Sociale zekerheid / bestuursrecht — nooit relevant voor privaatrechtelijk arbeidsgeschil
const SOCIAL_SECURITY_MARKERS =
  /\bbijstand\b|\bwwb\b|\bparticipatie(wet|uitkering)\b|\bwia\b|\bwajong\b|\baow\b|\bww\b|\bawbz\b|\bsvb\b|\buwv\b|terugvordering.{0,40}uitkering|intrekking.{0,40}uitkering|\bwao\b|\bziektewet\b|\bzw-uitkering\b/i;

const EMPLOYMENT_MARKERS =
  /ontslag|arbeidsovereenkomst|dienstverband|arbeidsrecht|loonvordering|\bcao\b|dringende reden|ernstig verwijtbaar/i;

function extractScoringTerms(query: string): string[] {
  return query
    .toLowerCase()
    .split(/\s+/)
    .map((w) => w.replace(/[^\p{L}]/gu, ''))
    .filter((w) => w.length > 3 && !FILLER_WORDS.has(w));
}

function scoreResult(result: SearchCaseLawResult, terms: string[]): number {
  const haystack = `${result.document_title} ${result.summary ?? ''}`.toLowerCase();
  const courtLower = result.court.toLowerCase();
  let score = 0;

  for (const term of terms) {
    if (haystack.includes(term)) score += 3;
  }

  const yr = ecliYear(result.ecli);
  if (yr >= 2022) score += 2;
  else if (yr >= 2019) score += 1;

  if (courtLower.includes('hoge raad')) score += 4;

  // Hard exclusions — irrelevant rechtsgebieden
  if (CRIMINAL_MARKERS.test(haystack)) score -= 20;
  if (SOCIAL_SECURITY_MARKERS.test(haystack)) score -= 20;
  // CRvB en RvS handelen bestuursrecht / sociale zekerheid — nooit privaatrechtelijk arbeidsrecht
  if (courtLower.includes('centrale raad')) score -= 25;
  if (courtLower.includes('raad van state')) score -= 25;
  if (courtLower.includes('college van beroep voor het bedrijfsleven')) score -= 15;

  // Positive signal: civil employment law
  if (EMPLOYMENT_MARKERS.test(haystack)) score += 6;

  return score;
}

function preFilterCases(cases: SearchCaseLawResult[], rawQuery: string): SearchCaseLawResult[] {
  // Hard filter: exclude non-civil courts by ECLI prefix — always wrong domain
  const civilOnly = cases.filter((r) => {
    if (isExcludedCourt(r.ecli)) {
      console.error(`[pre-filter] EXCLUDED (court) ${r.ecli}`);
      return false;
    }
    return true;
  });

  const terms = extractScoringTerms(rawQuery);
  const scored = civilOnly
    .map((r) => ({ r, score: scoreResult(r, terms) }))
    .filter(({ score }) => score > -5)
    .sort((a, b) => b.score - a.score);

  scored.forEach(({ r, score }) => console.error(`[pre-filter] ${r.ecli} (${r.court}) → ${score}`));
  return scored.slice(0, 15).map(({ r }) => r);
}

// ---------------------------------------------------------------------------
// STAGE 4 — Haiku re-ranking: JSON-only, temperature 0, strict system prompt
// ---------------------------------------------------------------------------

interface HaikuCasePick {
  ecli: string;
  titel: string;
  samenvatting: string;
  waarom_relevant: string;
}

async function rerankWithHaiku(
  query: string,
  cases: SearchCaseLawResult[],
): Promise<SearchCaseLawResult[]> {
  const model = process.env.ANTHROPIC_TITLE_MODEL;
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!model || !apiKey || cases.length === 0) {
    console.error('[haiku] env missing — returning first 3');
    return cases.slice(0, RESULT_CAP);
  }

  const casesList = cases
    .map(
      (c, i) =>
        `[${i + 1}] ECLI: ${c.ecli}\nTitel: ${c.document_title}\nRechtbank: ${c.court} | Datum: ${c.decision_date ?? 'onbekend'}\nSamenvatting: ${(c.summary ?? '').slice(0, 600)}`,
    )
    .join('\n\n');

  const systemPrompt = `Je bent een puur computergestuurd datafilter dat uitsluitend in JSON communiceert. Jouw taak is om uit de meegeleverde lijst van maximaal 20 rechtszaken de TOP 7 zaken te selecteren die qua juridische kern en feiten het dichtst bij de casus van de cliënt liggen. Sorteer op relevantie: de meest relevante zaak staat op positie 1.

CRUCIALE EN BINDENDE REGELS:
1. Je mag GEEN enkele menselijke zin typen. Antwoord NOOIT met inleidende of verklarende teksten zoals 'Helaas kan ik geen relevante uitspraken vinden...'.
2. Als er geen perfecte match is, selecteer dan de zaken die qua algemene juridische principes (zoals de hoorplicht, onderzoeksplicht of dringende reden) het meest waardevol zijn voor deze situatie. Je MOET altijd tot maximaal 7 zaken selecteren (minder is toegestaan als de pool kleiner is).
3. Je output moet DIRECT beginnen met het geopende bracket-teken [ en eindigen met ].
4. Gebruik GEEN markdown codeblocks (dus absoluut GEEN \`\`\`json ... \`\`\` in je response).

Het JSON-formaat moet exact zijn:
[
  {
    "ecli": "ECLI:NL:...",
    "titel": "Titel van de zaak",
    "samenvatting": "Korte officiële samenvatting",
    "waarom_relevant": "Korte juridische uitleg (max 2 zinnen) waarom deze zaak het beste aansluit."
  }
]`;

  const userPrompt = `VRAAG VAN GEBRUIKER:\n${query}\n\nBESCHIKBARE UITSPRAKEN:\n${casesList}`;

  console.error(`[haiku] re-ranking ${cases.length} cases → top ${RESULT_CAP} with model=${model}`);

  const response = await axios.post<{ content: Array<{ text: string }> }>(
    'https://api.anthropic.com/v1/messages',
    {
      model,
      max_tokens: 4000,
      temperature: 0,
      system: systemPrompt,
      messages: [{ role: 'user', content: userPrompt }],
    },
    {
      headers: {
        'x-api-key': apiKey,
        'anthropic-version': '2023-06-01',
        'content-type': 'application/json',
      },
      timeout: 20000,
    },
  );

  const text = response.data?.content?.[0]?.text ?? '';
  console.error(`[haiku] raw response length: ${text.length} chars`);

  try {
    const jsonMatch = text.match(/\[[\s\S]*\]/);
    if (!jsonMatch) throw new Error('no JSON array in response');

    const picks = JSON.parse(jsonMatch[0]) as HaikuCasePick[];
    const result = picks
      .slice(0, RESULT_CAP)
      .map((pick) => {
        const found = cases.find((c) => c.ecli === pick.ecli);
        if (!found) {
          console.error(`[haiku] ECLI not found in pool: ${pick.ecli}`);
          return null;
        }
        return { ...found, snippet: pick.waarom_relevant || found.snippet };
      })
      .filter((r): r is SearchCaseLawResult => r !== null);

    if (result.length === 0) throw new Error('none of the returned ECLIs matched the pool');

    console.error(`[haiku] selected: ${result.map((r) => r.ecli).join(', ')}`);
    return result;
  } catch (err) {
    console.error(`[haiku] JSON parse failed (${String(err)}) — falling back to first 3`);
    return cases.slice(0, RESULT_CAP);
  }
}

// ---------------------------------------------------------------------------
// Exported tool handler
// ---------------------------------------------------------------------------

export async function searchCaseLaw(
  db: Database,
  input: SearchCaseLawInput,
): Promise<ToolResponse<SearchCaseLawResult[]>> {
  const rawQuery = input.ecli ?? input.query ?? '';
  if (!rawQuery.trim()) {
    return { results: [], _metadata: generateResponseMetadata(db) };
  }

  const primaryQuery = await buildBooleanQuery(rawQuery);

  // Secondary query: strip case-specific keywords, keep only the first quoted legal phrase.
  // Casts a wider net to catch relevant cases that don't mention the exact fact terms.
  const phraseMatch = primaryQuery.match(/"([^"]+)"/);
  const secondaryQuery = phraseMatch
    ? `"${phraseMatch[1]}"`
    : (buildFallbackQuery(rawQuery) ?? primaryQuery);
  const runSecondary = secondaryQuery !== primaryQuery;

  try {
    // Stage 2 — two parallel Rechtspraak queries; merge + deduplicate by ECLI.
    const [primaryRaw, secondaryRaw] = await Promise.all([
      fetchFromRechtspraak(primaryQuery, input.legal_domain, input.date_from),
      runSecondary
        ? fetchFromRechtspraak(secondaryQuery, input.legal_domain, input.date_from)
        : Promise.resolve([] as SearchCaseLawResult[]),
    ]);

    const seen = new Set<string>();
    const raw = [...primaryRaw, ...secondaryRaw].filter((r) => {
      if (seen.has(r.ecli)) return false;
      seen.add(r.ecli);
      return true;
    });
    console.error(
      `[pipeline] merged: ${primaryRaw.length} + ${secondaryRaw.length} → ${raw.length} unique`,
    );

    // Stage 3 — fetch inhoudsindicatie for all results in parallel; drop those without content.
    const enriched = await Promise.all(
      raw.map(async (result) => {
        const content = await fetchInhoudsindicatie(result.ecli);
        if (content && content.length >= 30) {
          return {
            ...result,
            summary: content,
            snippet: content.slice(0, 200) + (content.length > 200 ? '…' : ''),
          };
        }
        // Fallback: behoud uitspraak met bestaande summary uit Atom-feed
        if (result.summary && result.summary.length >= 30) {
          return result;
        }
        return null;
      }),
    );
    const withContent = enriched.filter((r) => r !== null);
    console.error(`[pipeline] with inhoudsindicatie: ${withContent.length}/${raw.length}`);

    // Stage 3.5 — keyword pre-filter: score + keep top 12, drop criminal mismatches.
    const preFiltered = preFilterCases(withContent, rawQuery);
    console.error(`[pipeline] pre-filtered: ${preFiltered.length}`);

    // Stage 4 — Haiku picks top 3 based on original query + full summaries.
    const top3 = await rerankWithHaiku(rawQuery, preFiltered);

    console.error(`[pipeline] selected: ${top3.map((r) => r.ecli).join(', ')}`);

    return { results: addResultCitations(top3), _metadata: generateResponseMetadata(db) };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.error(`[searchCaseLaw] failed: ${message}`);
    return {
      results: [],
      _metadata: {
        ...generateResponseMetadata(db),
        note: `Rechtspraak.nl is tijdelijk niet bereikbaar (${message}). Probeer het later opnieuw.`,
      },
    };
  }
}
